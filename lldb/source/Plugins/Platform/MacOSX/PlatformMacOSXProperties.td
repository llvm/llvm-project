include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformdarwinkernel" in {
  def SearchForKexts: Property<"search-locally-for-kexts", "Boolean">,
    Global,
    DefaultTrue,
    Desc<"Automatically search for kexts on the local system when doing kernel debugging.">;
  def KextDirectories: Property<"kext-directories", "FileSpecList">,
    DefaultStringValue<"">,
    Desc<"Directories/KDKs to search for kexts in when starting a kernel debug session.">;
}
