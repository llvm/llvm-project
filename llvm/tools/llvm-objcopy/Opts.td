include "llvm/Option/OptParser.td"

multiclass Eq<string name> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>;
}

def help : Flag<["-", "--"], "help">;
defm binary_architecture : Eq<"binary-architecture">,
                           HelpText<"Used when transforming an architecture-less format (such as binary) to another format">;
def B : JoinedOrSeparate<["-"], "B">,
        Alias<binary_architecture>;
defm input_target : Eq<"input-target">,
                    HelpText<"Format of the input file">,
                    Values<"binary">;
defm output_target : Eq<"output-target">,
                     HelpText<"Format of the output file">,
                     Values<"binary">;
def O : JoinedOrSeparate<["-"], "O">,
        Alias<output_target>;
defm split_dwo : Eq<"split-dwo">,
                 MetaVarName<"dwo-file">,
                 HelpText<"Equivalent to extract-dwo on the input file to <dwo-file>, then strip-dwo on the input file">;
defm add_gnu_debuglink : Eq<"add-gnu-debuglink">,
                         MetaVarName<"debug-file">,
                         HelpText<"Add a .gnu_debuglink for <debug-file>">;
defm remove_section : Eq<"remove-section">,
                      MetaVarName<"section">,
                      HelpText<"Remove <section>">;
def R : JoinedOrSeparate<["-"], "R">,
        Alias<remove_section>;
defm keep : Eq<"keep">,
            MetaVarName<"section">,
            HelpText<"Keep <section>">;
defm only_keep : Eq<"only-keep">,
                 MetaVarName<"section">,
                 HelpText<"Remove all but <section>">;
def j : JoinedOrSeparate<["-"], "j">,
                      Alias<only_keep>;
defm add_section : Eq<"add-section">,
                   MetaVarName<"section=file">,
                   HelpText<"Make a section named <section> with the contents of <file>.">;
def strip_all : Flag<["-", "--"], "strip-all">,
                HelpText<"Remove non-allocated sections other than .gnu.warning* sections">;
def strip_all_gnu : Flag<["-", "--"], "strip-all-gnu">,
                    HelpText<"Compaitable with GNU objcopy's --strip-all">;
def strip_debug : Flag<["-", "--"], "strip-debug">,
                  HelpText<"Remove all debug information">;
def strip_dwo : Flag<["-", "--"], "strip-dwo">,
                HelpText<"Remove all DWARF .dwo sections from file">;
def strip_sections : Flag<["-", "--"], "strip-sections">,
                     HelpText<"Remove all section headers">;
def strip_non_alloc : Flag<["-", "--"], "strip-non-alloc">,
                      HelpText<"Remove all non-allocated sections">;
def extract_dwo : Flag<["-", "--"], "extract-dwo">,
                  HelpText<"Remove all sections that are not DWARF .dwo sections from file">;
def localize_hidden : Flag<["-", "--"], "localize-hidden">,
                      HelpText<"Mark all symbols that have hidden or internal visibility as local">;
