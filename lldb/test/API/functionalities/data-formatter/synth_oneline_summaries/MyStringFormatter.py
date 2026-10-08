import lldb


class MyStringSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self, max_num_children):
        return 0

    def has_children(self):
        return False
