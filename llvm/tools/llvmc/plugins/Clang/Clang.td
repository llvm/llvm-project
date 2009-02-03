// A replacement for the Clang's ccc script.
// Depends on the Base plugin.
// To compile, use this command:
//
//    cd $LLVMC2_DIR
//    make DRIVER_NAME=ccc2 BUILTIN_PLUGINS=Clang
//
// Or just use the default llvmc, which now has this plugin enabled.

include "llvm/CompilerDriver/Common.td"

def Priority : PluginPriority<1>;

def Options : OptionList<[
// Extern options
(switch_option "E", (extern)),
(switch_option "S", (extern)),
(switch_option "c", (extern)),
(switch_option "fsyntax-only", (extern)),
(switch_option "emit-llvm", (extern)),
(switch_option "pthread", (extern)),
(parameter_list_option "I", (extern)),
(parameter_list_option "include", (extern)),
(parameter_list_option "L", (extern)),
(parameter_list_option "l", (extern)),
(prefix_list_option "Wa,", (extern)),
(prefix_list_option "Wl,", (extern)),

(switch_option "clang", (help "Use Clang instead of llvm-gcc"))
]>;

class clang_based<string language, string cmd, string ext_E> : Tool<
[(in_language language),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case
           (switch_on "E"),
           (case
                (not_empty "o"),
                !strconcat(cmd, " -E $INFILE -o $OUTFILE"),
                (default),
                !strconcat(cmd, " -E $INFILE")),
          (and (switch_on "S"), (switch_on "emit-llvm")),
          !strconcat(cmd, " -emit-llvm $INFILE -o $OUTFILE"),
          (default),
          !strconcat(cmd, " -emit-llvm-bc $INFILE -o $OUTFILE"))),
 (actions (case (switch_on "E"),
                [(stop_compilation), (output_suffix ext_E)],
                (switch_on "fsyntax-only"), (stop_compilation),
                (and (switch_on "S"), (switch_on "emit-llvm")),
                           [(stop_compilation), (output_suffix "ll")],
                (and (switch_on "c"), (switch_on "emit-llvm")),
                           (stop_compilation),
                (not_empty "include"), (forward "include"),
                (not_empty "I"), (forward "I"))),
 (sink)
]>;

def clang_c : clang_based<"c", "clang -x c", "i">;
def clang_cpp : clang_based<"c++", "clang -x c++", "i">;
def clang_objective_c : clang_based<"objective-c",
    "clang -x objective-c", "mi">;
def clang_objective_cpp : clang_based<"objective-c++",
    "clang -x objective-c++", "mi">;

def as : Tool<
[(in_language "assembler"),
 (out_language "object-code"),
 (output_suffix "o"),
 (cmd_line "as $INFILE -o $OUTFILE"),
 (actions (case (not_empty "Wa,"), (unpack_values "Wa,"),
                (switch_on "c"), (stop_compilation)))
]>;

// Default linker
def llvm_ld : Tool<
[(in_language "object-code"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "llvm-ld -native -disable-internalize $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "pthread"), (append_cmd "-lpthread"),
          (not_empty "L"), (forward "L"),
          (not_empty "l"), (forward "l"),
          (not_empty "Wl,"), (unpack_values "Wl,"))),
 (join)
]>;

// Language map

def LanguageMap : LanguageMap<[
    LangToSuffixes<"c++", ["cc", "cp", "cxx", "cpp", "CPP", "c++", "C"]>,
    LangToSuffixes<"c", ["c"]>,
    LangToSuffixes<"objective-c", ["m"]>,
    LangToSuffixes<"c-cpp-output", ["i"]>,
    LangToSuffixes<"objective-c-cpp-output", ["mi"]>
]>;

// Compilation graph

def CompilationGraph : CompilationGraph<[
    OptionalEdge<"root", "clang_c",
                         (case (switch_on "clang"), (inc_weight))>,
    OptionalEdge<"root", "clang_cpp",
                         (case (switch_on "clang"), (inc_weight))>,
    OptionalEdge<"root", "clang_objective_c",
                         (case (switch_on "clang"), (inc_weight))>,
    OptionalEdge<"root", "clang_objective_cpp",
                         (case (switch_on "clang"), (inc_weight))>,
    Edge<"clang_c", "llc">,
    Edge<"clang_cpp", "llc">,
    Edge<"clang_objective_c", "llc">,
    Edge<"clang_objective_cpp", "llc">,
    OptionalEdge<"llc", "as", (case (switch_on "clang"), (inc_weight))>,
    Edge<"as", "llvm_ld">
]>;
