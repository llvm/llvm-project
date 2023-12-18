include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbollocatordebuginfod" in {
  def ServerURLs : Property<"server_urls", "Array">,
    ElementType<"String">,
    Desc<"An ordered list of Debuginfod server URLs to query for symbols. This defaults to the contents of the DEBUGINFOD_URLS environment variable.">;
}
