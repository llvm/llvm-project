include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "windows", Path = "platform.plugin.windows" in {
  def DisableDebugHeap: Property<"disable-debug-heap", "Boolean">,
    Global,
    DefaultTrue,
    Desc<"Specifies that the debug heap should not be used for debugging. "
    "The Windows debug heap has additional checks for heap related bugs at the cost of a noticeable performance penalty. "
    "By enabling this setting, _NO_DEBUG_HEAP=1 will be added to the environment.">;
}
