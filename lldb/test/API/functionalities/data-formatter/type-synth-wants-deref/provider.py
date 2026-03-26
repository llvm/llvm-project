class WrapperSynthProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.sum_value = None

    def update(self):
        self.sum_value = None
        ty = self.valobj.GetType()

        # Artificially bail out if LLDB passed us a reference or a pointer.
        if ty.IsPointerType() or ty.IsReferenceType():
            return False

        x = self.valobj.GetChildMemberWithName("x")
        y = self.valobj.GetChildMemberWithName("y")
        if x.IsValid() and y.IsValid():
            sum_val = x.GetValueAsUnsigned(0) + y.GetValueAsUnsigned(0)
            self.sum_value = self.valobj.CreateValueFromExpression("sum", str(sum_val))

        return False

    def num_children(self):
        return 1

    def get_child_at_index(self, index):
        if index == 0 and self.sum_value:
            return self.sum_value
        return None

    def get_child_index(self, name):
        if name == "sum":
            return 0
        return -1

    def has_children(self):
        return True
