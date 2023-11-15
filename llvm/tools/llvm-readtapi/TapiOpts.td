// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

class FF<string name, string help>: Flag<["-", "--"], name>, HelpText<help>;
multiclass JS<string name, string help, string var = ""> {
  def NAME #_EQ : Joined<["--"], name #"=">, HelpText<help>, MetaVarName<var>;
  def : Separate<["-", "--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

//
// General Driver options 
//
def help : FF<"help", "display this help">;
defm output: JS<"o", "write output to <file>","<file>">;

//
// Compare options
//
def compare : FF<"compare", "compare tapi files for library differences">;
