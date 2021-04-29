include "llvm/Option/OptParser.td"

def grp_generic : OptionGroup<"kind">, HelpText<"Generic Options">;

def help : Flag<["--"], "help">,
  HelpText<"Display available options (--help-hidden for more)">,
  Group<grp_generic>;

def help_hidden : Flag<["--"], "help-hidden">,
  Flags<[HelpHidden]>,
  HelpText<"Display all available options">,
  Group<grp_generic>;

def version : Flag<["--"], "version">,
  HelpText<"Display the version of this program">,
  Group<grp_generic>;



def grp_objdump : OptionGroup<"objdump">, HelpText<"llvm-objdump Options">;

def adjust_vma_EQ : Joined<["--"], "adjust-vma=">,
  MetaVarName<"offset">,
  HelpText<"Increase the displayed address by the specified offset">,
  Group<grp_objdump>;

def all_headers : Flag<["--"], "all-headers">,
  HelpText<"Display all available header information">,
  Group<grp_objdump>;
def : Flag<["-"], "x">, Alias<all_headers>, HelpText<"Alias for --all-headers">,
  Group<grp_objdump>;

def arch_name_EQ : Joined<["--"], "arch-name=">,
  HelpText<"Target arch to disassemble for, "
            "see --version for available targets">,
  Group<grp_objdump>;

def archive_headers : Flag<["--"], "archive-headers">,
  HelpText<"Display archive header information">,
  Group<grp_objdump>;
def : Flag<["-"], "a">, Alias<all_headers>,
  HelpText<"Alias for --archive-headers">,
  Group<grp_objdump>;

def demangle : Flag<["--"], "demangle">, HelpText<"Demangle symbol names">,
  Group<grp_objdump>;
def : Flag<["-"], "C">, Alias<demangle>, HelpText<"Alias for --demangle">,
  Group<grp_objdump>;

def disassemble : Flag<["--"], "disassemble">,
  HelpText<"Display assembler mnemonics for the machine instructions">,
  Group<grp_objdump>;
def : Flag<["-"], "d">, Alias<disassemble>, HelpText<"Alias for --disassemble">,
  Group<grp_objdump>;

def disassemble_all : Flag<["--"], "disassemble-all">,
  HelpText<"Display assembler mnemonics for the machine instructions">,
  Group<grp_objdump>;
def : Flag<["-"], "D">, Alias<disassemble_all>,
  HelpText<"Alias for --disassemble-all">,
  Group<grp_objdump>;

def symbol_description : Flag<["--"], "symbol-description">,
  HelpText<"Add symbol description for disassembly. This "
           "option is for XCOFF files only.">,
  Group<grp_objdump>;

def disassemble_symbols_EQ : Joined<["--"], "disassemble-symbols=">,
  HelpText<"List of symbols to disassemble. "
           "Accept demangled names when --demangle is "
           "specified, otherwise accept mangled names">,
  Group<grp_objdump>;

def disassemble_zeroes : Flag<["--"], "disassemble-zeroes">,
  HelpText<"Do not skip blocks of zeroes when disassembling">,
  Group<grp_objdump>;
def : Flag<["-"], "z">, Alias<disassemble_zeroes>,
  HelpText<"Alias for --disassemble-zeroes">,
  Group<grp_objdump>;

def disassembler_options_EQ : Joined<["--"], "disassembler-options=">,
  MetaVarName<"options">,
  HelpText<"Pass target specific disassembler options">,
  Group<grp_objdump>;
def : JoinedOrSeparate<["-"], "M">, Alias<disassembler_options_EQ>,
  HelpText<"Alias for --disassembler-options=">,
  Group<grp_objdump>;

def dynamic_reloc : Flag<["--"], "dynamic-reloc">,
  HelpText<"Display the dynamic relocation entries in the file">,
  Group<grp_objdump>;
def : Flag<["-"], "R">, Alias<dynamic_reloc>,
  HelpText<"Alias for --dynamic-reloc">,
  Group<grp_objdump>;

def dwarf_EQ : Joined<["--"], "dwarf=">,
  HelpText<"Dump of dwarf debug sections">, Values<"frames">,
  Group<grp_objdump>;

def fault_map_section : Flag<["--"], "fault-map-section">,
  HelpText<"Display contents of faultmap section">,
  Group<grp_objdump>;

def file_headers : Flag<["--"], "file-headers">,
  HelpText<"Display the contents of the overall file header">,
  Group<grp_objdump>;
def : Flag<["-"], "f">, Alias<file_headers>,
  HelpText<"Alias for --file-headers">,
  Group<grp_objdump>;

def full_contents : Flag<["--"], "full-contents">,
  HelpText<"Display the content of each section">,
  Group<grp_objdump>;
def : Flag<["-"], "s">, Alias<full_contents>,
  HelpText<"Alias for --full-contents">,
  Group<grp_objdump>;

def line_numbers : Flag<["--"], "line-numbers">,
  HelpText<"Display source line numbers with "
            "disassembly. Implies disassemble object">,
  Group<grp_objdump>;
def : Flag<["-"], "l">, Alias<line_numbers>,
  HelpText<"Alias for --line-numbers">,
  Group<grp_objdump>;

def macho : Flag<["--"], "macho">,
  HelpText<"Use MachO specific object file parser">,
  Group<grp_objdump>;
def : Flag<["-"], "m">, Alias<macho>, HelpText<"Alias for --macho">,
  Group<grp_objdump>;

def mcpu_EQ : Joined<["--"], "mcpu=">,
  MetaVarName<"cpu-name">,
  HelpText<"Target a specific cpu type (--mcpu=help for details)">,
  Group<grp_objdump>;

def mattr_EQ : Joined<["--"], "mattr=">,
  MetaVarName<"a1,+a2,-a3,...">,
  HelpText<"Target specific attributes (--mattr=help for details)">,
  Group<grp_objdump>;

def no_show_raw_insn : Flag<["--"], "no-show-raw-insn">,
  HelpText<"When disassembling instructions, "
           "do not print the instruction bytes.">,
  Group<grp_objdump>;

def no_leading_addr : Flag<["--"], "no-leading-addr">,
  HelpText<"Print no leading address">,
  Group<grp_objdump>;

def raw_clang_ast : Flag<["--"], "raw-clang-ast">,
  HelpText<"Dump the raw binary contents of the clang AST section">,
  Group<grp_objdump>;

def reloc : Flag<["--"], "reloc">,
  HelpText<"Display the relocation entries in the file">,
  Group<grp_objdump>;
def : Flag<["-"], "r">, Alias<reloc>, HelpText<"Alias for --reloc">,
  Group<grp_objdump>;

def print_imm_hex : Flag<["--"], "print-imm-hex">,
  HelpText<"Use hex format for immediate values">,
  Group<grp_objdump>;

def no_print_imm_hex : Flag<["--"], "no-print-imm-hex">,
  HelpText<"Do not use hex format for immediate values (default)">,
  Group<grp_objdump>;
def : Flag<["--"], "print-imm-hex=false">, Alias<no_print_imm_hex>,
  Group<grp_objdump>;

def private_headers : Flag<["--"], "private-headers">,
  HelpText<"Display format specific file headers">,
  Group<grp_objdump>;
def : Flag<["-"], "p">, Alias<private_headers>,
  HelpText<"Alias for --private-headers">,
  Group<grp_objdump>;

def section_EQ : Joined<["--"], "section=">,
  HelpText<"Operate on the specified sections only. "
           "With --macho dump segment,section">,
  Group<grp_objdump>;
def : Separate<["--"], "section">, Alias<section_EQ>,
  Group<grp_objdump>;
def : JoinedOrSeparate<["-"], "j">, Alias<section_EQ>,
  HelpText<"Alias for --section">,
  Group<grp_objdump>;

def section_headers : Flag<["--"], "section-headers">,
  HelpText<"Display summaries of the headers for each section.">,
  Group<grp_objdump>;
def : Flag<["--"], "headers">, Alias<section_headers>,
  HelpText<"Alias for --section-headers">,
  Group<grp_objdump>;
def : Flag<["-"], "h">, Alias<section_headers>,
  HelpText<"Alias for --section-headers">,
  Group<grp_objdump>;

def show_lma : Flag<["--"], "show-lma">,
  HelpText<"Display LMA column when dumping ELF section headers">,
  Group<grp_objdump>;

def source : Flag<["--"], "source">,
  HelpText<"Display source inlined with disassembly. Implies disassemble object">,
  Group<grp_objdump>;
def : Flag<["-"], "S">, Alias<source>, HelpText<"Alias for --source">,
  Group<grp_objdump>;

def start_address_EQ : Joined<["--"], "start-address=">,
  MetaVarName<"address">,
  HelpText<"Disassemble beginning at address">,
  Group<grp_objdump>;
def stop_address_EQ : Joined<["--"], "stop-address=">,
  MetaVarName<"address">,
  HelpText<"Stop disassembly at address">,
  Group<grp_objdump>;

def syms : Flag<["--"], "syms">,
  HelpText<"Display the symbol table">,
  Group<grp_objdump>;
def : Flag<["-"], "t">, Alias<syms>, HelpText<"Alias for --syms">,
  Group<grp_objdump>;

def symbolize_operands : Flag<["--"], "symbolize-operands">,
  HelpText<"Symbolize instruction operands when disassembling">,
  Group<grp_objdump>;

def dynamic_syms : Flag<["--"], "dynamic-syms">,
  HelpText<"Display the contents of the dynamic symbol table">,
  Group<grp_objdump>;
def : Flag<["-"], "T">, Alias<dynamic_syms>,
  HelpText<"Alias for --dynamic-syms">,
  Group<grp_objdump>;

def triple_EQ : Joined<["--"], "triple=">,
  HelpText<"Target triple to disassemble for, "
            "see --version for available targets">,
  Group<grp_objdump>;
def : Separate<["--"], "triple">,
  Alias<triple_EQ>,
  Group<grp_objdump>;

def unwind_info : Flag<["--"], "unwind-info">,
  HelpText<"Display unwind information">,
  Group<grp_objdump>;
def : Flag<["-"], "u">, Alias<unwind_info>,
  HelpText<"Alias for --unwind-info">,
  Group<grp_objdump>;

def wide : Flag<["--"], "wide">,
  HelpText<"Ignored for compatibility with GNU objdump">,
  Group<grp_objdump>;
def : Flag<["-"], "w">, Alias<wide>,
  Group<grp_objdump>;

def prefix : Separate<["--"], "prefix">,
  HelpText<"Add prefix to absolute paths">,
  Group<grp_objdump>;

def prefix_strip : Separate<["--"], "prefix-strip">,
  HelpText<"Strip out initial directories from absolute "
           "paths. No effect without --prefix">,
  Group<grp_objdump>;

def debug_vars_EQ : Joined<["--"], "debug-vars=">,
  Values<"unicode,ascii">,
  Group<grp_objdump>;
def : Flag<["--"], "debug-vars">,
  HelpText<"Print the locations (in registers or memory) of "
           "source-level variables alongside disassembly">,
  Alias<debug_vars_EQ>, AliasArgs<["unicode"]>,
  Group<grp_objdump>;

def debug_vars_indent_EQ : Joined<["--"], "debug-vars-indent=">,
  HelpText<"Distance to indent the source-level variable display, "
           "relative to the start of the disassembly">,
  Group<grp_objdump>;

def x86_asm_syntax_att : Flag<["--"], "x86-asm-syntax=att">,
  HelpText<"Emit AT&T-style disassembly">,
  Group<grp_objdump>;

def x86_asm_syntax_intel : Flag<["--"], "x86-asm-syntax=intel">,
  HelpText<"Emit Intel-style disassembly">,
  Group<grp_objdump>;

def mhvx : Flag<["--"], "mhvx">, HelpText<"Enable Hexagon Vector eXtensions">,
  Group<grp_objdump>;

def mhvx_v66 : Flag<["--"], "mhvx=v66">,
  Group<grp_objdump>;

def mv60 : Flag<["--"], "mv60">, HelpText<"Build for Hexagon V60">,
  Group<grp_objdump>;

def mv65 : Flag<["--"], "mv65">, HelpText<"Build for Hexagon V65">,
  Group<grp_objdump>;

def mv66 : Flag<["--"], "mv66">, HelpText<"Build for Hexagon V66">,
  Group<grp_objdump>;

def mv67 : Flag<["--"], "mv67">, HelpText<"Build for Hexagon V67">,
  Group<grp_objdump>;

def mv67t : Flag<["--"], "mv67t">, HelpText<"Build for Hexagon V67T">,
  Group<grp_objdump>;

def riscv_no_aliases : Flag<["--"], "riscv-no-aliases">,
  HelpText<"Disable the emission of assembler pseudo instructions">,
  Group<grp_objdump>;


def grp_mach_o : OptionGroup<"kind">, HelpText<"llvm-objdump MachO Specific Options">;

def private_header : Flag<["--"], "private-header">,
  HelpText<"Display only the first format specific file header">,
  Group<grp_mach_o>;

def exports_trie : Flag<["--"], "exports-trie">,
  HelpText<"Display mach-o exported symbols">,
  Group<grp_mach_o>;

def rebase : Flag<["--"], "rebase">,
  HelpText<"Display mach-o rebasing info">,
  Group<grp_mach_o>;

def bind : Flag<["--"], "bind">,
  HelpText<"Display mach-o binding info">,
  Group<grp_mach_o>;

def lazy_bind : Flag<["--"], "lazy-bind">,
  HelpText<"Display mach-o lazy binding info">,
  Group<grp_mach_o>;

def weak_bind : Flag<["--"], "weak-bind">,
  HelpText<"Display mach-o weak binding info">,
  Group<grp_mach_o>;

def g : Flag<["-"], "g">,
  HelpText<"Print line information from debug info if available">,
  Group<grp_mach_o>;

def dsym_EQ : Joined<["--"], "dsym=">,
  HelpText<"Use .dSYM file for debug info">,
  Group<grp_mach_o>;
def : Separate<["--"], "dsym">,
  Alias<dsym_EQ>,
  Group<grp_mach_o>;

def full_leading_addr : Flag<["--"], "full-leading-addr">,
  HelpText<"Print full leading address">,
  Group<grp_mach_o>;

def no_leading_headers : Flag<["--"], "no-leading-headers">,
  HelpText<"Print no leading headers">,
  Group<grp_mach_o>;

def universal_headers : Flag<["--"], "universal-headers">,
  HelpText<"Print Mach-O universal headers (requires --macho)">,
  Group<grp_mach_o>;

def archive_member_offsets : Flag<["--"], "archive-member-offsets">,
  HelpText<"Print the offset to each archive member for Mach-O archives "
           "(requires --macho and --archive-headers)">,
  Group<grp_mach_o>;

def indirect_symbols : Flag<["--"], "indirect-symbols">,
  HelpText<"Print indirect symbol table for Mach-O objects (requires --macho)">,
  Group<grp_mach_o>;

def data_in_code : Flag<["--"], "data-in-code">,
  HelpText<"Print the data in code table for Mach-O objects (requires --macho)">,
  Group<grp_mach_o>;

def function_starts : Flag<["--"], "function-starts">,
  HelpText<"Print the function starts table for "
           "Mach-O objects (requires --macho)">,
  Group<grp_mach_o>;

def link_opt_hints : Flag<["--"], "link-opt-hints">,
  HelpText<"Print the linker optimization hints for "
           "Mach-O objects (requires --macho)">,
  Group<grp_mach_o>;

def info_plist : Flag<["--"], "info-plist">,
  HelpText<"Print the info plist section as strings for "
           "Mach-O objects (requires --macho)">,
  Group<grp_mach_o>;

def dylibs_used : Flag<["--"], "dylibs-used">,
  HelpText<"Print the shared libraries used for linked "
           "Mach-O files (requires --macho)">,
  Group<grp_mach_o>;

def dylib_id : Flag<["--"], "dylib-id">,
  HelpText<"Print the shared library's id for the "
           "dylib Mach-O file (requires --macho)">,
  Group<grp_mach_o>;

def rpaths : Flag<["--"], "rpaths">,
  HelpText<"Print the runtime search paths for the "
           "Mach-O file (requires --macho)">,
  Group<grp_mach_o>;

def non_verbose : Flag<["--"], "non-verbose">,
  HelpText<"Print the info for Mach-O objects in non-verbose or "
           "numeric form (requires --macho)">,
  Group<grp_mach_o>;

def objc_meta_data : Flag<["--"], "objc-meta-data">,
  HelpText<"Print the Objective-C runtime meta data "
           "for Mach-O files (requires --macho)">,
  Group<grp_mach_o>;

def dis_symname : Separate<["--"], "dis-symname">,
  HelpText<"disassemble just this symbol's instructions (requires --macho)">,
  Group<grp_mach_o>;

def no_symbolic_operands : Flag<["--"], "no-symbolic-operands">,
  HelpText<"do not symbolic operands when disassembling (requires --macho)">,
  Group<grp_mach_o>;

def arch_EQ : Joined<["--"], "arch=">,
  HelpText<"architecture(s) from a Mach-O file to dump">,
  Group<grp_mach_o>;
def : Separate<["--"], "arch">,
  Alias<arch_EQ>,
  Group<grp_mach_o>;
