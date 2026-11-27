class ThingSynthetic:
    def __init__(self, valobj, _) -> None:
        self.valobj = valobj

    def num_children(self):
        return self.valobj.num_children

    def get_child_at_index(self, idx):
        return self.valobj.GetChildAtIndex(idx)

    # Use default implementation of get_child_index.


def __lldb_init_module(dbg, _):
    dbg.HandleCommand(f"type synthetic add -l {__name__}.ThingSynthetic Thing")
