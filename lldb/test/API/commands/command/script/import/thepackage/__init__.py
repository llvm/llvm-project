from . import TPunitA
from . import TPunitB


def __lldb_init_module(debugger, *args):
    debugger.HandleCommand(
        "command script add -f thepackage.TPunitA.command TPcommandA"
    )
    debugger.HandleCommand(
        "command script add -f thepackage.TPunitB.command TPcommandB"
    )
