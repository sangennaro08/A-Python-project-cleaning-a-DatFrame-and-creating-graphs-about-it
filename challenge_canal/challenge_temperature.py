import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


continua = True #per continuare il ciclo interno

base_dir = __file__.replace("challenge_temperature.py","")

#variabile speciale che restituisce il path di questo file

SEED = 0 #permette di modificare l'outcome delle temperature,impostabile come si vuole

mesi_ordinati = np.array(["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", 
                          "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"])

#caricamento dei dati,possibilità di inserire altri file della tipoligia data per il progetto
#in caso di errori si userà il file csv di default,che è quello fornito per il progetto

def load_data()->pd.DataFrame:

    dir = base_dir

    try:
        print("inserire il nome del file di cui ispezionare")
        print("NB:Inserire un file che abbia 2 colonne la prima Date con le date e la seconda Temp")
        print("ATTENZIONE ALL'ORTOGRAFIA")

        file=input(print("se usare il file di default clicare invio")) 

        if file == "":

            dir = dir + "daily-min-temperatures.csv"
            return pd.read_csv(dir)


        return pd.read_csv(dir+file)

    except FileNotFoundError:

        print("File non trovato inseriremo quello di default")
        dir = dir + "daily-min-temperatures.csv"
        return pd.read_csv(dir)


#fase 1:analisi dei dati e visualizzazione degli stessi,con controllo di qualità
#questo ci permetterà di decidere come adoperare per sporcare i dati nella fase soccessiva
#in base alla struttura del DF noi lo sporcheremo in modo adatto

def phase1(temp_data, temp_data_in_dates)->None:
    print("come primo punto è giusto inziare ordinare cronologicamente i dati in caso non lo siano già")
    temp_data = temp_data.sort_values(by='Date')
    print(temp_data.head())
    input()
    
    #inseriamo come indici le date
    print("per facilitare lo slicing inseriamo come indici le date e per la prima colonna la temperatura minima giornaliera")
    temp_data = temp_data.set_index("Date")
    print(temp_data)
    input(print("premere invio per vedere i controlli di qualità sui dati"))

    #informazioni principali
    print("\n\n\nprimi 5 dati del DF")
    print(temp_data.head())
    
    print("\n\n\nultimi 5 dati del DF")
    print(temp_data.tail())

    print("\n\n\ninformazioni sul DF")
    print(temp_data.info())

    print("\n\n\nstatistiche descrittive del DF")
    print(temp_data.describe())

    print("\n\n\ncontrollo dei dati mancanti")

    #controllo della quantità di nan nel DF
    if(temp_data.isna().sum().sum() == 0):
        print("non ci sono dati mancanti")
    else:
        print("ci sono dati mancanti,visto che sono tabelle di temperatura inseriremo 0 al posto di lasciarli")
        temp_data = temp_data.fillna(0)

    input(print("premere invio per vedere la visualizzazione dei dati"))

    return


#qui noi sporcheremo i dati intenzionalmente così da porterli ripulire in modo adatto
#in base alla struttura del DF noi lo sporcheremo in modo adatto,ad esempio inserendo dei dati mancanti o errati

def phase2(temp_data, temp_data_in_dates)->pd.DataFrame:

    #settiamo il SEED,si può decidere il numero sopra
    np.random.seed(SEED)

    dirty_DF = temp_data.copy() #creazione di una copia del DF originale per sporcare i dati senza modificare l'originale
    print("qua noi sperchermo 1/100 dei dati del DF inserendo dei dati NaN")

    index = np.random.choice(dirty_DF.index, size=dirty_DF.shape[0]//100, replace=False) #scelta casuale l' 1% dei dati da sporcare
    dirty_DF.loc[index, 'Temp'] = np.nan #inserimento di dati mancanti nei 10 indici scelti

    print("ora noi prenderemo una quantità casuale di righe e le reiseriremo dentro il DF stesso per poi mescolare tutto quanto")

    index = np.random.choice(dirty_DF.index, size=np.random.randint(1, dirty_DF.shape[0]//10), replace=False)
    dirty_DF = pd.concat([dirty_DF, dirty_DF.loc[index]], ignore_index=True) #inserimento di righe duplicate in posizioni casuali del DF

    dirty_DF = dirty_DF.sample(frac=1).reset_index(drop=True) #mescolamento del DF per rendere casuali le righe duplicate
    #dove
    #sample(frac=1) mescola TUTTO
    #reset_index(drop=true) resettiamo gli indici e diventano normali numeri

    print(dirty_DF)

    max_imp = 80
    min_imp = -50

    dirty_DF["Temp"] = dirty_DF["Temp"].apply(lambda x: max_imp if np.random.rand() < 0.01 else x) #inserimento di dati errati con una probabilità del 1%
    dirty_DF["Temp"] = dirty_DF["Temp"].apply(lambda x: min_imp if (np.random.rand() < 0.01 and x!=max_imp) else x) #inserimento di dati errati con una probabilità del 1%

    print(dirty_DF)
    input(print("premere invio per concludere la fase 2 e passare alla fase 3, che consiste nella pulizia dei dati sporchi"))

    dirty_DF = dirty_DF.set_index("Date")#risettiamo come indice le date

    return dirty_DF

#funzione per pulire i dati sporchi attraverso pandas,questa funzione dovrà essere in grado di identificare 
# e rimuovere i dati errati o mancanti inseriti nella fase 2,restituendo un DF pulito e pronto per l'analisi

def phase3(dirty_DF)->pd.DataFrame:


    clean_DF = dirty_DF.loc[~dirty_DF.index.duplicated(keep="first")] 
    #rimozione di tutte le righe duplicate, mantenendo solo quelle uniche
    #qua noi creiamo una maschera che identifica i duplicati tenendo la prima occorrenza
    #poi flippiamo così che il DF pulito ha i dati non duplicati

    clean_DF = clean_DF.sort_index()#riordiniamo gli indici
    print(clean_DF)
    input(print("premere invio per rimuovere i dati mancanti"))

    #controllo dei valori NaN nel DF
    print(f"NaN totali dentro il DF prima della rimozione dei dati errati: {clean_DF.isna().sum().sum()}")

    #qua noi portiamo gli indici ad essere trattati come date per poi riordinarli così da avere un DF ordinato

    clean_DF.index = pd.to_datetime(clean_DF.index)
    clean_DF = clean_DF.sort_index()

    #per evitare di dimenticae date usiamo questa funzione
    #infatti mancavano due date e siamo saliti a 3652 giornate
    all_dates = pd.date_range(start=clean_DF.index.min(), end=clean_DF.index.max(), freq="D")
    clean_DF = clean_DF.reindex(all_dates) #creazione di un intervallo di date completo tra la data minima e massima del DF

    #cerchiamo temperature impossibili con quantile per vedere quelli sottoe sopra la media del 1% e 99%
    low = clean_DF["Temp"].quantile(0.01)
    high = clean_DF["Temp"].quantile(0.99)

    mean_val = clean_DF["Temp"].mean()

    # maschera per identificare gli outlier PRIMA del clip
    mask = (clean_DF["Temp"] <= low) | (clean_DF["Temp"] >= high)

    # sostituisci gli outlier con la media

    clean_DF["Temp"] = clean_DF["Temp"].where(~mask, mean_val)
    #la funzione where permette di modificare certi valori per poi rimpiazzarli con uno a nostra scelta
    #in questo caso noi stiamo usando la maschera 
    #usiamo il tilde visto che where rimpiazza i false

    clean_DF["Temp"] = clean_DF["Temp"].ffill().bfill() #riempimento dei dati mancanti con il metodo di forward fill e backward fill
    print(clean_DF)

    print(f"NaN totali dentro il DF dopo la rimozione dei dati errati: {clean_DF.isna().sum().sum()}")
    input(print("premere invio per concludere la fase 3 e vedere il DF pulito"))

    return clean_DF

def phase4(clean_DF)->pd.DataFrame:

    giornata = {
        0: "Lunedì",
        1: "Martedì",
        2: "Mercoledì",
        3: "Giovedì",
        4: "Venerdì",
        5: "Sabato",
        6: "Domenica"
    }

    mesi = {
        1: "Gennaio",
        2: "Febbraio",
        3: "Marzo",
        4: "Aprile",
        5: "Maggio",
        6: "Giugno",
        7: "Luglio",
        8: "Agosto",
        9: "Settembre",
        10: "Ottobre",
        11: "Novembre",
        12: "Dicembre"
    }

    stagione = {
        1: "Inverno",
        2: "Inverno",
        3: "Primavera",
        4: "Primavera",
        5: "Primavera",
        6: "Estate",
        7: "Estate",
        8: "Estate",
        9: "Autunno",
        10: "Autunno",
        11: "autunno",#errore voluto per l'esercizio come richiesto
        12: "Inverno"
    }

    clean_DF = modify_clean_DF(clean_DF, giornata, mesi, stagione)

    print(clean_DF)

    cold_days =clean_DF[clean_DF["Fa freddo?"] == "Si"]
    print(f"i giorni in cui fa freddo sono: {cold_days['Temp'].describe()}")

    input(print("premere invio per concludere la fase 4 e vedere il DF modificato con le nuove colonne"))

    return  clean_DF

def modify_clean_DF(clean_DF, giornata, mesi, stagione)->pd.DataFrame:
   
    np.random.seed(SEED)
    clean_DF["Giorno_della_settimana"] = clean_DF.index.dayofweek.map(giornata)
    clean_DF["Numero_giorno"] = clean_DF.index.day
    clean_DF["Mese"] = clean_DF.index.month.map(mesi)
    clean_DF["Anno"] = clean_DF.index.year

    clean_DF["Stagione"] = clean_DF.index.month.map(stagione)
    print(clean_DF)

    input(print("premere invio per aggiustare la colonna della stagione"))

    print("notando un errore nel settare le stagioni,useremo replace")
    clean_DF["Stagione"] = clean_DF["Stagione"].replace("autunno", "Autunno")

    clean_DF["Fa freddo?"] = clean_DF["Temp"].apply(lambda x: "Si" if x < 10 else "No") #Aggiunta una colonna che dice se fa freddo o no
    #
    #se vuole per forza usare il boolean indexing come richiesto:
    #clean_DF["Fa fredddo?"] = clean_DF["Temp"] < 10
    #

    return clean_DF

#
#Questa funzione permette di manipolare il DF per creare una matrice che contiene media e std dei nostri dati
#
#
def phase5(clean_DF)->None:
        
        np.random.seed(SEED)
        grouped_by_months = clean_DF.groupby("Mese")["Temp"].agg(["mean","std"])
        print(grouped_by_months)

        #ordino il DF in base all'ordine dei mesi,così da avere una matrice ordinata e non ambigua

        grouped_by_months.index = pd.CategoricalIndex(grouped_by_months.index, categories=mesi_ordinati, ordered=True)
        grouped_by_months = grouped_by_months.sort_index()

        print(grouped_by_months)

        input(print("premere invio per creare una matrice numpy con i dati del DF pulito"))

        #creazione matrice
        matrice = grouped_by_months.to_numpy()
        print(matrice)
        print(matrice.shape)

        bootstrap(clean_DF)
        
        #l'esercizio per creare una matrice numpy con i dati del DF pulito è stato completato,ma c'è un problema...
        #NON SI PUò FARE il compito "wow" dal modo richiesto dalla traccia!!!
        #il DF originale anche se è precisamente 3650 righe e abbiamo per ogni anno 365 perfetti per fare la matrice ha una pecca...
        #Mancano 2 giorni,il 31-12-1984 e 31-12-1988 impedendo di fare una matrice con forma non ambigua,infatti con la pulizia dei dati
        #sono stati aggiunti i giorni mancanti e fare una matrice così renderebbe la forma ambigua e non accettabile
        
        #posso fare però un pivot table e rimpiazzare i dati NaN degli anni non bisestili

        clean_DF2 = clean_DF.copy()

        clean_DF2["Giorno_anno"] = clean_DF2.index.dayofyear

        matrice = clean_DF2.pivot(

            values = "Temp",
            index = "Anno",
            columns = "Giorno_anno"

        )

        matrice = matrice.fillna("Data non esistente")

        print(matrice)

        return

def bootstrap(clean_DF)->None:


    np.random.seed(SEED)
    print("adesso faremo il bootstraping per stimare la media e la deviazione standard della temperatura minima giornaliera")

    array = np.random.choice(clean_DF["Temp"], size = len(clean_DF["Temp"]), replace = True)
    print(f"media campione: {array.mean()} e percentile 95%: {np.percentile(array, 95)}")

    input(print("premere invio per concludere la fase 5 e passare alla fase 6, che consiste nella visualizzazione dei dati attraverso grafici"))
    
    return

def phase6(clean_DF)->None:

    if input(print("voler vedere i grafici assieme cliccare 1 altrimenti cliccare un tasto quasiasi per vederli separati") == "1"):

        figure, axis = plt.subplots(2,2)
        setting_plot_1(axis,clean_DF)
        setting_plot_2(axis,clean_DF)
        setting_plot_3(axis,clean_DF)
        setting_plot_4(axis,clean_DF)

        print("vuoi salvare i grafici creati? (si/no)")
        if input() == "si":
            print("SALVATAGGIO GRAFICO IN FORMATO PNG")
            
            figure.savefig(base_dir + "csv_files_and_graphs/4_graphs.png",dpi=300)

        plt.tight_layout()
        plt.show()
        plt.close()

    else:

        axis = plt.subplot()

        setting_plot_1_second(axis,clean_DF)
        
        axis = plt.subplot()

        setting_plot_2_second(axis,clean_DF)
        
        axis = plt.subplot()

        setting_plot_3_second(axis,clean_DF)
        
        axis = plt.subplot()

        setting_plot_4_second(axis,clean_DF)
    
    return

def setting_plot_1(axis,clean_DF)->None:

    temp = clean_DF.groupby("Anno")["Temp"].mean()

    axis[0,0].set_title("Temperatura media annua", fontsize = 7)
    
    axis[0,0].plot(temp.index, temp.values,
                        marker = "o",ms = 8, mfc = "black",color = "black", 
                        linestyle = "dashed",linewidth = 2)

    axis[0,0].set_ylabel("Temperatura in °C")
    axis[0,0].set_xlabel("Anni")
    axis[0,0].set_xticks(range(clean_DF.index.year.min(),clean_DF.index.year.max() + 1))
    axis[0,0].set_xticklabels(range(clean_DF.index.year.min(), clean_DF.index.year.max() +1), rotation = 45, fontsize = 5)
    axis[0,0].set_xlim(clean_DF.index.year.min(),clean_DF.index.year.max())

    axis[0,0].tick_params(axis = "both", labelsize = 7)
    axis[0,0].grid(True, linestyle = "--", color = "gray", axis = "y")

    return

def setting_plot_2(axis,clean_DF)->None:  

    axis[0,1].set_title("Temperatura minima giornaliera,quante volte compaiono?", fontsize = 7)
    axis[0,1].hist(clean_DF["Temp"], bins=clean_DF.shape[0]//5, edgecolor="none")

    axis[0,1].set_xlabel("Temperature in °C", fontsize = 5)
    axis[0,1].set_ylabel("Quantità di comparsa di una temperatura" ,fontsize = 5)

    return

def setting_plot_3(axis,clean_DF)->None:

    dati = [clean_DF[clean_DF.index.month == m]["Temp"].values for m in range(1, 13)]
    axis[1,0].boxplot(dati,sym="")
    axis[1,0].set_title("Temperatura media mensile", fontsize = 7)

    axis[1,0].set_xticks(range(1,13))
    axis[1,0].set_xticklabels(mesi_ordinati, rotation=90, fontsize = 5)

    axis[1,0].set_ylabel("Temperatura in °C", fontsize = 5)

    return

def setting_plot_4(axis,clean_DF)->None:
    
    dati = clean_DF.pivot_table(values="Temp", columns="Anno", index="Mese", aggfunc="mean")
    
    # riordina le righe in base all'ordine cronologico dei mesi
    dati.index = pd.CategoricalIndex(dati.index, categories=mesi_ordinati, ordered=True)
    dati = dati.sort_index()
    
    axis[1,1].imshow(dati, aspect='auto', cmap='coolwarm', clim=(clean_DF["Temp"].min(), clean_DF["Temp"].max()))
    axis[1,1].set_title("temperatura con heatmap", fontsize = 7)

    axis[1,1].set_yticks(range(12))
    axis[1,1].set_yticklabels(mesi_ordinati, fontsize = 5)

    axis[1,1].set_xticks(range((clean_DF.index.year.max() + 1) - clean_DF.index.year.min()))
    axis[1,1].set_xticklabels(range(clean_DF.index.year.min(), (clean_DF.index.year.max() + 1)), rotation=90)

    return

def setting_plot_1_second(axis, clean_DF)->None:


    axis.plot(clean_DF.groupby("Anno")["Temp"].mean().index, clean_DF.groupby("Anno")["Temp"].mean().values,
              marker = "o",ms = 8, mfc = "black",color = "black", linestyle = "dashed",linewidth = 2)

    axis.set_title("temperatura media Annua")

    axis.set_ylabel("Temperatura in °C")
    axis.set_xlabel("Anni")

    axis.set_xticks(range(clean_DF.index.year.min(),clean_DF.index.year.max() + 1))
    axis.set_xticklabels(clean_DF.index.year.unique(), rotation=90)
    axis.set_xlim(clean_DF.index.year.min(),clean_DF.index.year.max())

    axis.grid(True, linestyle = "--", color = "gray", axis = "y")

    print("salvataggio immagine")
    print("SALVATAGGIO GRAFICO IN FORMATO PNG")
    plt.savefig(base_dir + "csv_files_and_graphs/grafico_linee.png")

    plt.show()
    plt.close()

    return

def setting_plot_2_second(axis, clean_DF)->None:

    axis.hist(clean_DF["Temp"], bins=clean_DF.shape[0]//5, color = "black", edgecolor="none",linewidth = 0)
    axis.set_title("Temperatura minima giornaliera,quante volte compaiono?")
    axis.set_ylabel("Quantità di comparsa di una temperatura")
    axis.set_xlabel("Temperature in °C")

    print("salvataggio immagine")
    print("SALVATAGGIO GRAFICO IN FORMATO PNG")
    plt.savefig(base_dir + "csv_files_and_graphs/istogramma.png")

    plt.show()
    plt.close()

    return

def setting_plot_3_second(axis, clean_DF)->None:

    dati = [clean_DF[clean_DF.index.month == m]["Temp"].values for m in range(1, 13)]
    axis.boxplot(dati,sym="",showmeans = True)
    axis.set_title("Temperatura media mensile")
    axis.set_ylabel("Temperatura in °C")
    axis.set_xticks(range(1,13))
    axis.set_xticklabels(mesi_ordinati, rotation=90)

    print("salvataggio immagine")
    print("SALVATAGGIO GRAFICO IN FORMATO PNG")
    plt.savefig(base_dir + "csv_files_and_graphs/boxplot.png")

    plt.show()
    plt.close()

    return

def setting_plot_4_second(axis, clean_DF)->None:            

    dati = clean_DF.pivot_table(values="Temp", columns="Anno", index="Mese", aggfunc="mean")
    
    # riordina le righe in base all'ordine cronologico dei mesi
    dati.index = pd.CategoricalIndex(dati.index, categories=mesi_ordinati, ordered=True)
    dati = dati.sort_index()

    axis.imshow(dati, aspect='auto', cmap='coolwarm', clim=(clean_DF["Temp"].min(), clean_DF["Temp"].max()))
    axis.set_title("Temperatura media per mese,mostrando quelli più caldi")

    axis.set_yticks(range(12))
    axis.set_yticklabels(mesi_ordinati)

    axis.set_xticks(range((clean_DF.index.year.max() + 1) - clean_DF.index.year.min()))
    axis.set_xticklabels(range(clean_DF.index.year.min(), (clean_DF.index.year.max() + 1)), rotation=90)
    axis.set_xlabel("Anni")

    print("salvataggio immagine")
    print("SALVATAGGIO GRAFICO IN FORMATO PNG")
    plt.savefig(base_dir + "csv_files_and_graphs/heatmap.png")

    plt.show()
    plt.close()

    return

def save_DF(clean_DF)->None:

    print("SALVATAGGIO DF IN FORMATO CSV")
    final_DF.to_csv(base_dir + "csv_files_and_graphs/cleaned_daily_min_temperatures.csv", index=True)

    print("SALVATAGGIO COMPLETATO CON SUCCCESSO!")

    return

while(continua):

    temp_data=load_data()
    print(temp_data)
    temp_data_in_dates = pd.to_datetime(temp_data['Date'])
    print(temp_data_in_dates)

    print("inizieremo la fase 1 del progetto, che consiste nell'analisi dei dati e nella visualizzazione degli stessi")
    print("in questa fase partiremo mostrando la struttura,dati e con adatto controllo di qualità")
    phase1(temp_data, temp_data_in_dates)

    print("la fase 2 consiste nello sporcare i dati,ovvero inserire dei dati errati o mancanti per poi doverli pulire nella fase 3")
    dirty_DF=phase2(temp_data, temp_data_in_dates)

    print("la fase 3 consiste nella pulizia dei dati sporchi attraverso pandas")
    clean_DF=phase3(dirty_DF)

    print("inzia la fase 4, modifica del DF inserendo colonne per giorni,mesi e anni delle date con piccola incoerenza da risolvere")
    final_DF = phase4(clean_DF)

    print("adesso inizia la fase 5,creeremo una matrice di numpy con i dati del DF pulito")
    phase5(clean_DF)

    print("iniziamo adesso la creazione dei grafici per visualizzare i dati")
    phase6(clean_DF)

        
    print("adesso decidi cosa fare")
    print("0)uscire senza salvare il DF ripulito")
    print("1)uscire salvando il DF ripulito")
    print("2)continuare il programma salvando il DF utilizzato")
    print("3)continuare senza salvare il DF utilizzato")

    match input():

        case "0":
            continua = False
            break

        case "1":
            continua = False
            save_DF(clean_DF)
            break

        case "2":
            save_DF(clean_DF)
            continue

        case _:
            continue

print("grazie per aver usato il programma,arrivederci!")