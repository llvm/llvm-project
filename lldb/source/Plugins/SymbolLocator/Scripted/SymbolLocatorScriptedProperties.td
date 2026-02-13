include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbollocatorscripted" in {
  def ScriptClass : Property<"script-class", "String">,
    DefaultStringValue<"">,
    Desc<"The name of the Python class that implements the scripted symbol locator. The class should implement methods like locate_source_file(module_spec, original_source_file) to resolve source files.">;
}
