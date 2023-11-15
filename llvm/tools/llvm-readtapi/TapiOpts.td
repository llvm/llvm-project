// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

class FF<string name, string help>: Flag<["-", "--"], name>, HelpText<help>;
class JS<string name, string help, string var = ""> : JoinedOrSeparate<["-", "--"], name>, HelpText<help>, MetaVarName<var>;

//
// General Driver options 
//
def help : FF<"help", "display this help">;
def output: JS<"o", "write output to <file>","<file>">;

//
// Compare options
//
def compare : FF<"compare", "compare tapi files for library differences">;
