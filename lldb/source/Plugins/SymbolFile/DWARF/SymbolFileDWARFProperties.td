include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbolfiledwarf" in {
  def SymLinkPaths: Property<"comp-dir-symlink-paths", "FileSpecList">,
    Global,
    DefaultStringValue<"">,
    Desc<"If the DW_AT_comp_dir matches any of these paths the symbolic links will be resolved at DWARF parse time.">;
  def IgnoreIndexes: Property<"ignore-file-indexes", "Boolean">,
    Global,
    DefaultFalse,
    Desc<"Ignore indexes present in the object files and always index DWARF manually.">;
}
