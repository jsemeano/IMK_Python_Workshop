from auto_nutzer.auto import Auto 

class SportAuto(Auto):

    """
    SubClass von Auto.
    Wir brauchen können ein __init__, falls wir mehr Parameter wollen
    """

    def __init__(self,motor,tuer,farbe):
        super().__init__(motor,tuer)
        self.farbe = farbe
    
    def preis_(self):
        self.preis = self.motor*30+self.tuer*400
        return self.preis