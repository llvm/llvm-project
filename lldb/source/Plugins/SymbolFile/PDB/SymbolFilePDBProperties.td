include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbolfilepdb" in {
  def UseNativeReader: Property<"use-native-reader", "Enum">,
    Global,
    DefaultEnumValue<"eUseNativePDBReaderDefault">,
    EnumValues<"OptionEnumValues(g_native_pdb_reader_enums)">,
    Desc<"When 'on', use the native PDB reader based on LLVM's PDB support as opposed to the reader using Microsoft's DIA SDK. "
         "On Windows, the default is controlled by the LLDB_USE_NATIVE_PDB_READER environment variable. "
         "If this is set, then the native reader is used. "
         "Note that the setting value will always have priority and that it needs to be set before a target is created. "
         "On other platforms, the native reader is always used.">;
}
