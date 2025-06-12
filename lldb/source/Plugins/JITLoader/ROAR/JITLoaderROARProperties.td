include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "jitloaderroar" in {
  def Enable: Property<"enable", "Enum">,
    Global,
    DefaultEnumValue<"eEnableJITLoaderROARDefault">,
    EnumValues<"OptionEnumValues(g_enable_jit_loader_roar_enumerators)">,
    Desc<"Enable ROAR's JIT compilation interface (default: enabled on 64-bit systems)">;

  def ROARSharedLibraryPath: Property<"so-path", "FileSpec">,
    Global,
    DefaultStringValue<"">,
    Desc<"Path to the libroar-runtime.so used to generate debug info for pre-compiled functions">;

  def ROARLLDBShimsSharedLibraryPath: Property<"shims-so-path", "FileSpec">,
    Global,
    DefaultStringValue<"">,
    Desc<"Path to the libroar-lldb-shims.so used to parse ROAR's Compiler/Runtime ABI data">;

  def EagerSymbolGeneration: Property<"eager-symbol-generation", "Boolean">,
    DefaultFalse,
    Desc<"Generate symbols as soon as jitcode is registered with LLDB.">;

  def DisableTrampolineStop: Property<"disable-trampoline-stop", "Boolean">,
    DefaultTrue,
    Desc<"Disable breakpoints stopping in a trampoline code.">;

  def TestShimsErrorCreation: Property<"test-shims-error-creation", "Boolean">,
    DefaultFalse,
    Desc<"Generates error in shims. Used for testing only.">;
}
