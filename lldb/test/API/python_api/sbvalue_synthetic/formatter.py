import lldb


class FooSyntheticProvider:
    def __init__(self, valobj, dict):
        target = valobj.GetTarget()
        data = lldb.SBData.CreateDataFromCString(lldb.eByteOrderLittle, 8, "S")
        self._child = valobj.CreateValueFromData(
            "synth_child", data, target.GetBasicType(lldb.eBasicTypeChar)
        )

    def num_children(self):
        return 1

    def get_child_at_index(self, index):
        if index != 0:
            return None
        return self._child

    def get_child_index(self, name):
        if name == "synth_child":
            return 0
        return None
