class PairProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def num_children(self):
        return 1

    def get_child_at_index(self, index):
        return self.valobj.GetChildMemberWithName("second")
