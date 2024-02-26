class Auto:

    def __init__(self,motor,tuer):
        """
        Class constructor: Dieser Method ist Pflicht in alle Klassen.
        __init__ wird aufgeruft jedes Mal eine neue Instanz von der Klass ergestellt wird.
        Es definiert die Parameter, die für die Klass benötigt werden
        """
        self.motor = motor # Attribut: Daten, die sich auf die spezifisch erstellte Instanz beziehen
        self.tuer = tuer


    def preis_(self):
        """
        Class method: Eine Funktion, die der Klass Instanz modifiziert
        In diesem Fall, wie werden der Motor und Tür Attribute nutzen, um ein Preis zu kalkulieren
        """
        self.preis = self.motor*10+self.tuer*200
        return self.preis

    def print_preis_(self):
        print(f"Ein Auto mit einem {self.motor} cc Motor und {self.tuer} Türe kostet {self.preis}€.")