include "llvm/Option/OptParser.td"

multiclass Eq<string name> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>;
}

def help : Flag<["-", "--"], "help">;

defm output : Eq<"o">,
              MetaVarName<"output">,
              HelpText<"Write output to <file>">;

def strip_debug : Flag<["-", "--"], "strip-debug">,
                  HelpText<"Remove debugging symbols only">;

defm remove_section : Eq<"remove-section">,
                      MetaVarName<"section">,
                      HelpText<"Remove <section>">;

def R : JoinedOrSeparate<["-"], "R">,
        Alias<remove_section>;

defm keep_symbol : Eq<"keep-symbol">,
                   MetaVarName<"symbol">,
                   HelpText<"Do not remove symbol <symbol>">;

def K : JoinedOrSeparate<["-"], "K">,
        Alias<keep_symbol>;
