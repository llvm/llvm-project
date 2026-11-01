import lldb


class TaggedSyntheticProvider:
    """A custom synthetic child provider that produces a single child that
    does not depend on reading the tagged pointer's (nonexistent) backing
    memory, so it works for tagged pointers.
    """

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def num_children(self):
        return 1

    def get_child_index(self, name):
        return 0 if name == "synthetic_child" else -1

    def get_child_at_index(self, index):
        if index != 0:
            return lldb.SBValue()
        # Build the child from raw data rather than by reading the object, so it
        # works for a tagged pointer (which has no backing memory).
        target = self.valobj.GetTarget()
        int_type = target.GetBasicType(lldb.eBasicTypeInt)
        data = lldb.SBData.CreateDataFromSInt32Array(
            target.GetByteOrder(), target.GetAddressByteSize(), [42]
        )
        return self.valobj.CreateValueFromData("synthetic_child", data, int_type)

    def update(self):
        return False
