import lldb


def derives_from_base(sbtype, internal_dict):
    for base in sbtype.get_bases_array():
        if base.GetName() == "Base":
            return True
    return False


class SynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        return 1

    def get_child_index(self, name):
        return 0

    def get_child_at_index(self, index):
        if index == 0:
            return self.valobj.CreateValueFromExpression("synthetic_child", "9999")
        return None


def register_formatters(debugger):
    cat = debugger.CreateCategory("callback_formatters")
    cat.AddTypeSummary(
        lldb.SBTypeNameSpecifier(
            "formatters_with_callback.derives_from_base", lldb.eFormatterMatchCallback
        ),
        lldb.SBTypeSummary.CreateWithScriptCode("return 'hello from callback summary'"),
    )
    cat.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier(
            "formatters_with_callback.derives_from_base", lldb.eFormatterMatchCallback
        ),
        lldb.SBTypeSynthetic.CreateWithClassName(
            "formatters_with_callback.SynthProvider"
        ),
    )
    cat.SetEnabled(True)
