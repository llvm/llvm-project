include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;

def grp_general : OptionGroup<"Dsymutil">, HelpText<"Dsymutil Options">;

def help: F<"help">,
  HelpText<"Prints this help output.">,
  Group<grp_general>;
def: Flag<["-"], "h">,
  Alias<help>,
  HelpText<"Alias for --help">,
  Group<grp_general>;

def version: F<"version">,
  HelpText<"Prints the dsymutil version.">,
  Group<grp_general>;
def: Flag<["-"], "v">,
  Alias<version>,
  HelpText<"Alias for --version">,
  Group<grp_general>;

def verbose: F<"verbose">,
  HelpText<"Enable verbose mode.">,
  Group<grp_general>;

def verify: F<"verify">,
  HelpText<"Run the DWARF verifier on the linked DWARF debug info.">,
  Group<grp_general>;

def no_output: F<"no-output">,
  HelpText<"Do the link in memory, but do not emit the result file.">,
  Group<grp_general>;

def no_swiftmodule_timestamp: F<"no-swiftmodule-timestamp">,
  HelpText<"Don't check timestamp for swiftmodule files.">,
  Group<grp_general>;

def no_odr: F<"no-odr">,
  HelpText<"Do not use ODR (One Definition Rule) for type uniquing.">,
  Group<grp_general>;

def dump_debug_map: F<"dump-debug-map">,
  HelpText<"Parse and dump the debug map to standard output. Not DWARF link will take place.">,
  Group<grp_general>;

def yaml_input: F<"y">,
  HelpText<"Treat the input file is a YAML debug map rather than a binary.">,
  Group<grp_general>;

def papertrail: F<"papertrail">,
  HelpText<"Embed warnings in the linked DWARF debug info.">,
  Group<grp_general>;

def assembly: F<"S">,
  HelpText<"Output textual assembly instead of a binary dSYM companion file.">,
  Group<grp_general>;

def symtab: F<"symtab">,
  HelpText<"Dumps the symbol table found in executable or object file(s) and exits.">,
  Group<grp_general>;
def: Flag<["-"], "s">,
  Alias<symtab>,
  HelpText<"Alias for --symtab">,
  Group<grp_general>;

def flat: F<"flat">,
  HelpText<"Produce a flat dSYM file (not a bundle).">,
  Group<grp_general>;
def: Flag<["-"], "f">,
  Alias<flat>,
  HelpText<"Alias for --flat">,
  Group<grp_general>;

def minimize: F<"minimize">,
  HelpText<"When used when creating a dSYM file with Apple accelerator tables, "
           "this option will suppress the emission of the .debug_inlines, "
           ".debug_pubnames, and .debug_pubtypes sections since dsymutil "
           "has better equivalents: .apple_names and .apple_types. When used in "
           "conjunction with --update option, this option will cause redundant "
           "accelerator tables to be removed.">,
  Group<grp_general>;
def: Flag<["-"], "z">,
  Alias<minimize>,
  HelpText<"Alias for --minimize">,
  Group<grp_general>;

def update: F<"update">,
  HelpText<"Updates existing dSYM files to contain the latest accelerator tables and other DWARF optimizations.">,
  Group<grp_general>;
def: Flag<["-"], "u">,
  Alias<update>,
  HelpText<"Alias for --update">,
  Group<grp_general>;

def output: Separate<["--", "-"], "o">,
  MetaVarName<"<filename>">,
  HelpText<"Specify the output file. Defaults to <input file>.dwarf">,
  Group<grp_general>;
def: Separate<["--", "-"], "out">,
  Alias<output>,
  HelpText<"Alias for --o">,
  Group<grp_general>;
def: Joined<["--", "-"], "out=">, Alias<output>;
def: Joined<["--", "-"], "o=">, Alias<output>;

def oso_prepend_path: Separate<["--", "-"], "oso-prepend-path">,
  MetaVarName<"<path>">,
  HelpText<"Specify a directory to prepend to the paths of object files.">,
  Group<grp_general>;
def: Joined<["--", "-"], "oso-prepend-path=">, Alias<oso_prepend_path>;

def symbolmap: Separate<["--", "-"], "symbol-map">,
  MetaVarName<"<bcsymbolmap>">,
  HelpText<"Updates the existing dSYMs inplace using symbol map specified.">,
  Group<grp_general>;
def: Joined<["--", "-"], "symbol-map=">, Alias<symbolmap>;

def arch: Separate<["--", "-"], "arch">,
  MetaVarName<"<arch>">,
  HelpText<"Link DWARF debug information only for specified CPU architecture"
           "types. This option can be specified multiple times, once for each"
           "desired architecture. All CPU architectures will be linked by"
           "default.">,
  Group<grp_general>;
def: Joined<["--", "-"], "arch=">, Alias<arch>;

def accelerator: Separate<["--", "-"], "accelerator">,
  MetaVarName<"<accelerator type>">,
  HelpText<"Specify the desired type of accelerator table. Valid options are 'Apple', 'Dwarf' and 'Default'">,
  Group<grp_general>;
def: Joined<["--", "-"], "accelerator=">, Alias<accelerator>;

def toolchain: Separate<["--", "-"], "toolchain">,
  MetaVarName<"<toolchain>">,
  HelpText<"Embed toolchain information in dSYM bundle.">,
  Group<grp_general>;

def threads: Separate<["--", "-"], "num-threads">,
  MetaVarName<"<threads>">,
  HelpText<"Specifies the maximum number of simultaneous threads to use when linking multiple architectures.">,
  Group<grp_general>;
def: Separate<["-"], "j">,
  Alias<threads>,
  HelpText<"Alias for --num-threads">,
  Group<grp_general>;

def remarks_prepend_path: Separate<["--", "-"], "remarks-prepend-path">,
  MetaVarName<"<path>">,
  HelpText<"Specify a directory to prepend to the paths of the external remark files.">,
  Group<grp_general>;
def: Joined<["--", "-"], "remarks-prepend-path=">, Alias<remarks_prepend_path>;

def remarks_output_format: Separate<["--", "-"], "remarks-output-format">,
  MetaVarName<"<format>">,
  HelpText<"Specify the format to be used when serializing the linked remarks.">,
  Group<grp_general>;
def: Joined<["--", "-"], "remarks-output-format=">, Alias<remarks_output_format>;
