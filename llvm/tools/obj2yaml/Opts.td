include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--"], name #"=">, HelpText<help>;
  def : Separate<["--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def help : FF<"help", "Display this help">;
def : F<"h", "Alias for --help">, Alias<help>;
def o : JoinedOrSeparate<["-"], "o">, HelpText<"Output filename">, MetaVarName<"<filename>">;
defm raw_segment : Eq<"raw-segment", "Mach-O: dump the raw contents of the listed segment (data or linkedit) instead of parsing it">, MetaVarName<"<segment>">;
def version : FF<"version", "Display the version">;
