import lldb


class myArraySynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        size_valobj = self.valobj.GetChildMemberWithName("m_arr_size")
        if size_valobj:
            return size_valobj.GetValueAsUnsigned(0)
        return 0

    def get_child_at_index(self, index):
        size_valobj = self.valobj.GetChildMemberWithName("m_arr_size")
        arr = self.valobj.GetChildMemberWithName("m_array")
        if not size_valobj or not arr:
            return None
        max_idx = size_valobj.GetValueAsUnsigned(0)
        if index >= max_idx:
            return None
        return arr.GetChildAtIndex(index)
