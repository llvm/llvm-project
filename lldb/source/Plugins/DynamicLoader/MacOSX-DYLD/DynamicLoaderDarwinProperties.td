include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "dynamicloaderdarwin" in {
  def EnableParallelImageLoad: Property<"enable-parallel-image-load", "Boolean">,
    Global,
    DefaultFalse,
    Desc<"Load images in parallel.">;
}
