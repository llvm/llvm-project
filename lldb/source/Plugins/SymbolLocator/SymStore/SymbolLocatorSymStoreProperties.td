include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbollocatorsymstore", Path = "plugin.symbol-locator.symstore" in {
  def SymStoreURLs : Property<"urls", "Array">,
    ElementType<"String">,
    Desc<"List of local symstore directories to query for symbols">;
}
