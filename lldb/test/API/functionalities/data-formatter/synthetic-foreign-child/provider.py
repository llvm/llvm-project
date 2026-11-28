import lldb


class ForeignChildProvider:
    """Returns a new root value for its child and keeps no reference to it."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def num_children(self):
        return 1

    def get_child_index(self, name):
        return 0 if name == "child" else None

    def get_child_at_index(self, index):
        value = self.valobj.GetChildMemberWithName("value")
        return self.valobj.GetTarget().CreateValueFromData(
            "child", value.GetData(), value.GetType()
        )


class ChildOfAnotherValueProvider:
    """Returns a child of a root value that the provider keeps alive."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.holder = None

    def num_children(self):
        return 1

    def get_child_index(self, name):
        return 0 if name == "value" else None

    def get_child_at_index(self, index):
        inner = self.valobj.GetChildMemberWithName("inner")
        self.holder = (
            self.valobj.GetTarget()
            .CreateValueFromData("holder", inner.GetData(), inner.GetType())
            .GetNonSyntheticValue()
        )
        return self.holder.GetChildMemberWithName("value")
