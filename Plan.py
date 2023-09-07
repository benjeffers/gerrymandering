class Plan:
    def __init__(self, districts, size, pop) -> None:
        self.districts = districts
        self.size = size
        self.pop = pop 

    def get_size(self):
        return sum(self.size)
    
    def get_pop(self):
        return sum(self.pop)