include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FB<string name, string help> : Flag<["-","--"], name>, HelpText<help>;
class JoinedOrSep<string name, string help>
    : JoinedOrSeparate<["-","--"], name>, HelpText<help>;

def help : FB<"help", "Display available options">;
def : F<"h", "Alias for --help">, Alias<help>;

def version : FB<"version", "Display the version">;
def : F<"V", "Alias for --version">, Alias<version>;

def override : JoinedOrSep<"override",
                           "Input bitcode file which can override previously "
                           "defined symbols">,
                MetaVarName<"filename">;
def import : JoinedOrSep<"import",
                         "Pair of function name and filename, where function "
                         "should be imported from bitcode in filename">,
             MetaVarName<"function:filename">;
def summary_index : JoinedOrSep<"summary-index",
                                "Module summary index filename">,
                    MetaVarName<"filename">;

def output : JoinedOrSep<"output", "Override output filename">,
             MetaVarName<"filename">;
def o : JoinedOrSeparate<["-"], "o">, Alias<output>;

def internalize : FB<"internalize", "Internalize linked symbols">;
def disable_debug_info_type_map
    : FB<"disable-debug-info-type-map",
         "Don't use a uniquing type map for debug info">;
def only_needed : FB<"only-needed", "Link only needed symbols">;
def f : F<"f", "Enable binary output on terminals">;
def disable_lazy_loading
    : FB<"disable-lazy-loading", "Disable lazy module loading">;
def S : F<"S", "Write output as LLVM assembly">;
def v : F<"v", "Print information about actions taken">;
def d : F<"d", "Print assembly as linked">, Flags<[HelpHidden]>;
def suppress_warnings
    : FB<"suppress-warnings", "Suppress all linking warnings">;
def disable_verify
    : FB<"disable-verify", "Do not run the verifier">, Flags<[HelpHidden]>;
def ignore_non_bitcode
    : FB<"ignore-non-bitcode",
         "Do not report an error for non-bitcode files in archives">,
      Flags<[HelpHidden]>;
