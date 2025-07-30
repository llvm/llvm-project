include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "language_cplusplus" in {
  def FunctionNameFormat: Property<"function-name-format", "FormatEntity">,
    Global,
    DefaultStringValue<"${function.return-left}${function.scope}${ansi.fg.yellow}${function.basename}${ansi.normal}${function.template-arguments}${function.formatted-arguments}${function.return-right}${function.qualifiers}${function.suffix}">,
    Desc<"C++ specific frame format string to use when displaying stack frame information for threads.">;
}
