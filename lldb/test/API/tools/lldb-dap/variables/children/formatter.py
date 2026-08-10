import lldb


num_children_calls = []


class TestSyntheticProvider:
    def __init__(self, valobj, dict):
        target = valobj.GetTarget()
        self._type = valobj.GetType()
        data = lldb.SBData.CreateDataFromCString(lldb.eByteOrderLittle, 8, "S")
        name = "child" if "Not" in self._type.GetName() else "[0]"
        self._child = valobj.CreateValueFromData(
            name, data, target.GetBasicType(lldb.eBasicTypeChar)
        )

    def num_children(self):
        num_children_calls.append(self._type.GetName())
        return 1

    def get_child_at_index(self, index):
        if index != 0:
            return None
        return self._child

    def get_child_index(self, name):
        if name == self._child.GetName():
            return 0
        return None


def __lldb_init_module(debugger, dict):
    cat = debugger.CreateCategory("TestCategory")
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("Indexed"),
        lldb.SBTypeSynthetic.CreateWithClassName("formatter.TestSyntheticProvider"),
    )
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier("NotIndexed"),
        lldb.SBTypeSynthetic.CreateWithClassName("formatter.TestSyntheticProvider"),
    )
    cat.SetEnabled(True)
