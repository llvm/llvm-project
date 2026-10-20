include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--"], name #"=">, HelpText<help>;
  def : Separate<["--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def D : JoinedOrSeparate<["-"], "D">, HelpText<"Define the specified macro to the specified definition">, MetaVarName<"<macro>=<definition>">;
defm docnum : Eq<"docnum", "Read the specified document from the input (default = 1)">, MetaVarName<"<n>">;
def E : F<"E", "Just print the preprocessed file">;
def help : FF<"help", "Display this help">;
def : F<"h", "Alias for --help">, Alias<help>;
defm max_size : Eq<"max-size", "Set the maximum allowed output size (0 means no limit) [ELF and COFF only]">, MetaVarName<"<bytes>">;
def o : JoinedOrSeparate<["-"], "o">, HelpText<"Output filename">, MetaVarName<"<filename>">;
def version : FF<"version", "Display the version">;
