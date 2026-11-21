include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

def help : FF<"help", "Display this help">;
def : F<"h", "Alias for --help">, Alias<help>;
def leading_lines : FF<"leading-lines", "Preserve line numbers">;
def no_leading_lines : FF<"no-leading-lines", "Don't preserve line numbers (default)">;
def version : FF<"version", "Display the version">;
