include "llvm/Option/OptParser.td"

multiclass Eq<string name> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>;
}

def help : Flag<["-", "--"], "help">;

defm output : Eq<"o">,
              MetaVarName<"output">,
              HelpText<"Write output to <file>">;

def strip_all : Flag<["-", "--"], "strip-all">,
                HelpText<"Remove non-allocated sections other than .gnu.warning* sections">;

def strip_debug : Flag<["-", "--"], "strip-debug">,
                  HelpText<"Remove debugging symbols only">;

def d : Flag<["-"], "d">,
        Alias<strip_debug>;

def g : Flag<["-"], "g">,
        Alias<strip_debug>;

def S : Flag<["-"], "S">,
        Alias<strip_debug>;

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

def discard_all : Flag<["-", "--"], "discard-all">,
                  HelpText<"Remove all local symbols except file and section symbols">;
def x : Flag<["-"], "x">,
        Alias<discard_all>;

def strip_unneeded : Flag<["-", "--"], "strip-unneeded">,
                      HelpText<"Remove all symbols not needed by relocations">;
