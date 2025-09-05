include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbolfilepdb" in {
  def Reader: Property<"reader", "Enum">,
    Global,
    DefaultEnumValue<"ePDBReaderDefault">,
    EnumValues<"OptionEnumValues(g_pdb_reader_enums)">,
    Desc<"Selects the reader for PDB symbol files. "
         "The native PDB reader that uses LLVM's PDB support is always available (value: 'native'). "
         "Secondly, the DIA PDB reader is only available if LLVM was comppiled with Microsoft's DIA SDK on Windows (value: 'DIA'). "
         "By default, the DIA PDB reader is used if available. "
         "The LLDB_USE_NATIVE_PDB_READER environment variable can be used to switch to the native reader when this setting has the default value. "
         "Otherwise, the setting always has priority.">;
}
