# ErkennungVonObjekteMitYoloV5
Erkennung Von Objekte Mit YoloV5

Erkennung von Objekten in Bildern/Videos

Es gibt verschiedene Algorithmen zur Objekterkennung wie **YOLO (You Only Look Once)**, **Single Shot Detector (SSD)**, **Faster R-CNN**, **Histogram of Oriented Gradients (HOG)** usw.

**YOLO (You Only Look Once)** ist ein einzigartiger Algorithmus zur Objekterkennung, bei dem das gesamte Bild nur einmal betrachtet wird, um Objekte zu erkennen. Es ist schneller als die meisten anderen Algorithmen, da es das gesamte Bild auf einmal analysiert, anstatt es in kleinere Regionen aufzuteilen.

**Single Shot Detector (SSD)** ist ein weiterer Algorithmus zur Objekterkennung, der darauf abzielt, schnell und präzise Ergebnisse zu liefern. Es nutzt ein einzelnes Neuronale Netzwerk, um sowohl die Position als auch die Klassifizierung von Objekten vorherzusagen.

**Faster R-CNN** ist ein weiteres bekanntes Verfahren zur Objekterkennung. Es nutzt Region Proposals, um potenzielle Objekte im Bild zu identifizieren und dann ein tiefes Neuronales Netzwerk, um die Objekte genau zu klassifizieren.

**Histogram of Oriented Gradients (HOG)** ist ein anderer Algorithmus zur Objekterkennung, der hauptsächlich für die Erkennung von Personen verwendet wird. Es nutzt die Merkmale der Gradientenrichtungen, um die Form eines Objekts zu beschreiben und ermöglicht so die Erkennung von Personen auch bei unterschiedlichen Positionen und Orientierungen.

Es gibt noch viele andere Algorithmen, die für die Objekterkennung verwendet werden können, wie z.B. **RetinaNet**, **Mask R-CNN**, **U-Net**, etc. Jeder Algorithmus hat seine eigenen Stärken und Schwächen und es hängt von der Anwendung und den Anforderungen ab, welcher am besten geeignet ist.

In diesem Repo werde ich Yolo-V5 verwenden, um ein benutzerdefiniertes Objekterkennungsmodell zu trainieren. YOLO ist eines der bekanntesten Objekterkennungsmodelle und wird häufig in Anwendungen wie der Überwachung, der Autonomen Navigation und der Bildanalyse verwendet. Es ist bekannt dafür, dass es schnell und präzise ist und in Echtzeit ausgeführt werden kann.


## Vorwissen:
**Deep Learning** ist eine Methode des maschinellen Lernens, die auf neuronalen Netzwerken basiert. Es nutzt tiefe Schichten von Neuronen, um komplexe Muster in Daten zu erkennen und zu lernen.

**Computer Vision** ist ein Bereich der Informatik, der sich mit der Verarbeitung und Analyse von Bild- und Videosignalen befasst. Es nutzt Algorithmen und Methoden aus dem maschinellen Lernen, um Bilder und Videos automatisch zu analysieren und zu verstehen.

**Deep Learning in der Computer Vision** nutzt tiefe neuronale Netze, um komplexe Muster in Bildern und Videos zu erkennen. Es wird verwendet, um Aufgaben wie die Erkennung von Gesichtern, Objekten, Texten und Bewegungen zu lösen. Einige der bekanntesten Anwendungen von Deep Learning in der Computer Vision sind die Erkennung von Objekten und Gesichtern, die Überwachung von Verkehrsströmen, die Automatisierung von Wartungsprozessen und die Generierung von Bildern.

Es gibt verschiedene Arten von tiefen neuronalen Netzen, die in der Computer Vision verwendet werden, wie z.B. Konvolutionale Neuronale Netze (CNNs), Recurrent Neural Networks (RNNs) und Generative Adversarial Networks (GANs). Jede Art von Netzwerk hat ihre eigenen Stärken und Schwächen und es hängt von der Anwendung und den Anforderungen ab, welches Netzwerk am besten geeignet ist.

## Schritte
Dies sind die Schritte, die in diesem Repo zur Erstellung eines eigenen benutzerdefinierten Objekterkennungsmodells mit YOLOv5 behandelt werden:

1.  Vorbereitung des Datensatzes: Sammeln und Vorbereiten der Bilder und Labels für die Objekte, die erkannt werden sollen.
    
2.  Umgebungsaufbau: Installieren der Abhängigkeiten für YOLOv5, wie z.B. Python, PyTorch und OpenCV.
    
3.  Einrichten der Daten und Verzeichnisse: Anlegen der Ordnerstruktur für die Daten und die Ausgaben des Trainings.
    
4.  Einrichten der YAML-Dateien für das Training: Erstellen der Konfigurationsdateien für das Training des Modells.
    
5.  Training des Modells: Verwenden des Trainingsdatensatzes, um das Modell zu trainieren und die Gewichte des Modells zu optimieren.
    
6.  Auswertung des Modells: Überprüfen der Leistung des Modells anhand von Metriken wie Genauigkeit und Fehlerrate.
    
7.  Visualisierung der Trainingsdaten: Anzeigen von Beispielbildern und deren vorhergesagten Ergebnissen, um die Leistung des Modells zu verstehen.
    
8.  Ausführen der Inferenz auf Testbildern: Anwenden des trainierten Modells auf neue, unbekannte Bilder, um die Erkennung von Objekten zu testen.
    
9.  Exportieren der Gewichtsdateien für spätere Verwendung: Speichern der Gewichtsdateien des trainierten Modells, um sie später in anderen Anwendungen zu verwenden.


## Vorbereitung des Datensatzes
Ich werde meine eigenen Bilder verwenden und mithilfe von Annotationstools wie LabelImg, CVAT oder Online-Webseite wie makesense.ai die Objekte im Bild markieren und beschriften.

Eine weitere Möglichkeit ist die Verwendung von bereits vorhandenen öffentlichen Datensätzen wie [https://public.roboflow.com/object-detection](https://public.roboflow.com/object-detection), wo man verschiedene Datensätze finden kann, die für das Training des Modells verwendet werden können.

Nachdem der Datensatz vorbereitet ist, können Sie mit dem Aufbau der Umgebung und dem Training des Modells fortfahren.


## Umgebungsaufbau
Wir werde mit Google Colab arbeiten. Google Colab ist ein nützliches Tool, da es eine kostenlose GPU für 12 Stunden bereitstellt und es einfach ist, die Umgebung einzurichten. Um es zu verwenden, benötigen Sie lediglich ein Google-Konto.
Hier ist die Umgebeung von Yolov5 github Repo: https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb

1. GitHub-Repository von YoloV5 klonen
![image](https://user-images.githubusercontent.com/48016716/214070598-0c7281e0-29cb-4c71-b649-d0069b5a8758.png)

2. vortrainierte Daten hochladen
![image](https://user-images.githubusercontent.com/48016716/214071180-e07dcfa7-0342-4c1e-8876-3866cf5e6212.png)

3. Hochgeladene Daten unzipen
![image](https://user-images.githubusercontent.com/48016716/214071975-78b932c2-d86e-4362-9d4a-92baf2294d4e.png)


## Das Modell trainieren
Einfach den Befehl hier ausführen und schon sind haben wir ein vortrainiertes Model 
![image](https://user-images.githubusercontent.com/48016716/214072817-84fe7431-3326-4d47-aaa2-49e2965ec5ed.png)


Die vortrainierte Daten findet man hier
![image](https://user-images.githubusercontent.com/48016716/214073275-efbead9a-e15b-4c79-ad63-d2985ce7431b.png)


