include "llvm/Option/OptParser.td"

def help : Flag<["-", "--"], "help">;
def h : Flag<["-"], "h">, Alias<help>;

def version : Flag<["-", "--"], "version">,
              HelpText<"Print the version and exit.">;

def action_group : OptionGroup<"action group">;

def verify_arch
    : Option<["-", "--"], "verify_arch", KIND_REMAINING_ARGS>,
      Group<action_group>,
      HelpText<
          "Verify that the specified arch_types are present in the input file">;

def archs : Option<["-", "--"], "archs", KIND_FLAG>,
            Group<action_group>,
            HelpText<"Display the arch_types present in the input file">;

def thin : Option<["-", "--"], "thin", KIND_SEPARATE>,
           Group<action_group>,
           HelpText<"Create a thin output file of specified arch_type from the "
                    "fat input file. Requires -output option">;

def output : Option<["-", "--"], "output", KIND_SEPARATE>,
             HelpText<"Create output file with specified name">;
def o : JoinedOrSeparate<["-"], "o">, Alias<output>;
