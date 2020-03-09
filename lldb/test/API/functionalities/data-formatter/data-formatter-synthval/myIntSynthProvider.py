class myIntSynthProvider(object):

    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.val = self.valobj.GetChildMemberWithName("theValue")

    def num_children(self):
        return 0

    def get_child_at_index(self, index):
        return None

    def get_child_index(self, name):
        return None

    def update(self):
        return False

    def has_children(self):
        return False

    def get_value(self):
        return self.val


class myArraySynthProvider(object):

    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.array = self.valobj.GetChildMemberWithName("array")

    def num_children(self, max_count):
        if 16 < max_count:
            return 16
        return max_count

    def get_child_at_index(self, index):
        return None  # Keep it simple when this is not tested here.

    def get_child_index(self, name):
        return None  # Keep it simple when this is not tested here.

    def has_children(self):
        return True
