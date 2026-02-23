include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbollocatormicrosoft", Path = "plugin.symbol-locator.microsoft" in {
  def SymStoreURLs : Property<"symstore-urls", "Array">,
    ElementType<"String">,
    Desc<"An ordered list of symstore URLs to query for symbols. This is prepended to the contents of the _NT_SYMBOL_PATH environment variable.">;
}
