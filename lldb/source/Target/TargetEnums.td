include "../../include/lldb/Core/EnumsBase.td"

def dynamic_value_types : EnumDef<"", [
  EnumValue<"eNoDynamicValues", "no-dynamic-values", "Don't calculate the dynamic type of values">,
  EnumValue<"eDynamicCanRunTarget", "run-target", "Calculate the dynamic type of values even if you have to run the target.">,
  EnumValue<"eDynamicDontRunTarget", "no-run-target", "Calculate the dynamic type of values, but don't run the target.">,
]>;

def inline_breakpoint_enums : EnumDef<"eInlineBreakpoints", [
  EnumValue<"Never", "never",
    "Never look for inline breakpoint locations (fastest). "
    "This setting should only be used if you know that no inlining occurs in "
    "your programs.">,
  EnumValue<"Headers", "headers",
    "Only check for inline breakpoint locations when setting breakpoints in "
    "header files, but not when setting breakpoint in implementation source "
    "files (default).">,
  EnumValue<"Always", "always",
    "Always look for inline breakpoint locations when setting file and line "
    "breakpoints (slower but most accurate).">,
]>;

def x86_dis_flavor_value_types : EnumDef<"eX86DisFlavor", [
  EnumValue<"Default", "default", "Disassembler default (currently att).">,
  EnumValue<"Intel", "intel", "Intel disassembler flavor.">,
  EnumValue<"ATT", "att", "AT&T disassembler flavor.">,
]>;

def jit_engine_value_types : EnumDef<"eJITEngine", [
  EnumValue<"MCJIT", "mcjit", "Use LLVM's MCJIT execution engine.">,
  EnumValue<"ORC", "orc", "Use LLVM's ORC execution engine.">,
]>;

def import_std_module_value_types : EnumDef<"eImportStdModule", [
  EnumValue<"False", "false",
    "Never import the 'std' C++ module in the expression parser.">,
  EnumValue<"Fallback", "fallback",
    "Retry evaluating expressions with an imported 'std' C++ module if they "
    "failed to parse without the module. This allows evaluating more complex "
    "expressions involving C++ standard library types.">,
  EnumValue<"True", "true",
    "Always import the 'std' C++ module. This allows evaluating more complex "
    "expressions involving C++ standard library types. This feature is "
    "experimental.">,
]>;

def dynamic_class_info_helper_value_types : EnumDef<"eDynamicClassInfoHelper", [
  EnumValue<"Auto", "auto",
    "Automatically determine the most appropriate method for the target OS.">,
  EnumValue<"RealizedClassesStruct", "RealizedClassesStruct",
    "Prefer using the realized classes struct.">,
  EnumValue<"CopyRealizedClassList", "CopyRealizedClassList",
    "Prefer using the CopyRealizedClassList API.">,
  EnumValue<"GetRealizedClassList", "GetRealizedClassList",
    "Prefer using the GetRealizedClassList API.">,
]>;

def hex_immediate_style_values : EnumDef<"Disassembler::eHexStyle", [
  EnumValue<"C", "c", "C-style (0xffff).">,
  EnumValue<"Asm", "asm", "Asm-style (0ffffh).">,
]>;

def load_script_from_sym_file_values : EnumDef<"eLoadScriptFromSymFile", [
  EnumValue<"True", "true", "Load debug scripts inside symbol files">,
  EnumValue<"False", "false", "Do not load debug scripts inside symbol files.">,
  EnumValue<"Warn", "warn",
    "Warn about debug scripts inside symbol files but do not load them.">,
  EnumValue<"Trusted", "trusted",
    "Load debug scripts inside trusted symbol files, and warn about scripts "
    "from untrusted symbol files.">,
]>;

def load_cwd_lldbinit_values : EnumDef<"eLoadCWDlldbinit", [
  EnumValue<"True", "true", "Load .lldbinit files from current directory">,
  EnumValue<"False", "false",
    "Do not load .lldbinit files from current directory">,
  EnumValue<"Warn", "warn",
    "Warn about loading .lldbinit files from current directory">,
]>;

def memory_module_load_level_values : EnumDef<"eMemoryModuleLoadLevel", [
  EnumValue<"Minimal", "minimal",
    "Load minimal information when loading modules from memory. "
    "Currently this setting loads sections only.">,
  EnumValue<"Partial", "partial",
    "Load partial information when loading modules from memory. "
    "Currently this setting loads sections and function bounds.">,
  EnumValue<"Complete", "complete",
    "Load complete information when loading modules from memory. "
    "Currently this setting loads sections and all symbols.">,
]>;
