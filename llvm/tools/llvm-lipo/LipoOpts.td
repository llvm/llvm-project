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
