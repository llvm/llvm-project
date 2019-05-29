include "llvm/Option/OptParser.td"

def help : Flag<["-", "--"], "help">;
def h : Flag<["-"], "h">, Alias<help>;

def version : Flag<["-", "--"], "version">,
              HelpText<"Print the version and exit.">;

def verify_arch : Option<["-", "--"], "verify_arch", KIND_REMAINING_ARGS>,
                  HelpText<"Verify that the specified arch_types are present in the input file">;
