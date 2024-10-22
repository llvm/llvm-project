import lldb


class fooSynthProvider:
    # For testing purposes, we'll keep track of the maximum value of
    # max_num_children we've been called with.
    MAX_NUM_CHILDREN_MAX = 0

    @classmethod
    def reset_max_num_children_max(cls):
        old_value = fooSynthProvider.MAX_NUM_CHILDREN_MAX
        fooSynthProvider.MAX_NUM_CHILDREN_MAX = 0
        return old_value

    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.int_type = valobj.GetType().GetBasicType(lldb.eBasicTypeInt)

    def num_children(self, max_num_children):
        fooSynthProvider.MAX_NUM_CHILDREN_MAX = max(
            fooSynthProvider.MAX_NUM_CHILDREN_MAX, max_num_children
        )
        return 3

    def get_child_at_index(self, index):
        if index == 0:
            child = self.valobj.GetChildMemberWithName("a")
        if index == 1:
            child = self.valobj.CreateChildAtOffset("fake_a", 1, self.int_type)
        if index == 2:
            child = self.valobj.GetChildMemberWithName("r")
        return child

    def get_child_index(self, name):
        if name == "a":
            return 0
        if name == "fake_a":
            return 1
        return 2
