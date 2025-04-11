include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "language_cplusplus" in {
  def FrameFormat: Property<"frame-format", "FormatEntity">,
    Global,
    DefaultStringValue<"${function.return-left}${function.scope}${ansi.fg.cyan}${function.basename}${ansi.normal}${function.template-arguments}${function.arguments}${function.return-right}${function.qualifiers}">,
    Desc<"C++ specific frame format string to use when displaying stack frame information for threads.">;
}
