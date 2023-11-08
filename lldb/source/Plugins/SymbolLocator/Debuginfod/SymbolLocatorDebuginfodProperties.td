include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbollocatordebuginfod" in {
  def URLs: Property<"urls", "String">,
    Global,
    DefaultStringValue<"">,
    Desc<"A space-separated, ordered list of Debuginfod server URLs to query for symbols">;
}
