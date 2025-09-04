include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "language_swift" in {
  def FunctionNameFormat: Property<"function-name-format", "FormatEntity">,
    Global,
    DefaultStringValue<"${function.prefix}${ansi.fg.yellow}${function.basename}${ansi.normal}${function.name-qualifiers}${function.template-arguments}${function.formatted-arguments}${function.suffix}">,
    Desc<"Swift specific frame format string to use when displaying stack frame information for threads.">;
}