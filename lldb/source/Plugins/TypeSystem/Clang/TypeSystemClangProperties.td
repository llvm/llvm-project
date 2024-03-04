include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "typesystemclang" in {
  def RedeclCompletion: Property<"experimental-redecl-completion", "Boolean">,
    Global,
    DefaultFalse,
    Desc<"Use redeclarations to complete Clang types. Experimental setting.">;
}
