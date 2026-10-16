include "../../../../include/lldb/Core/EnumsBase.td"

def enable_jit_loader_gdb_enumerators : EnumDef<"eEnableJITLoaderGDB", [
  EnumValue<"Default", "default",
    "Enable JIT compilation interface for all platforms except macOS">,
  EnumValue<"On", "on", "Enable JIT compilation interface">,
  EnumValue<"Off", "off", "Disable JIT compilation interface">,
]>;
