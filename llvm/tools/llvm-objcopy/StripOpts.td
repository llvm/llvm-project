include "llvm/Option/OptParser.td"

multiclass Eq<string name> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>;
}

def help : Flag<["-", "--"], "help">;

def strip_debug : Flag<["-", "--"], "strip-debug">,
                  HelpText<"Remove debugging symbols only">;

