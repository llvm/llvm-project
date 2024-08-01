include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;

// General options
def generic_group : OptionGroup<"Genric Options">, HelpText<"Generic Options">;
def help : F<"help", "Display this help">, Group<generic_group>;
def : Flag<["-"], "h">, Alias<help>, HelpText<"Alias for --help">, Group<generic_group>;
def version : F<"version", "Display the version">, Group<generic_group>;
def : Flag<["-"], "v">, Alias<version>, HelpText<"Alias for --version">, Group<generic_group>;

// Action options
def action_group : OptionGroup<"Action">, HelpText<"Action (required)">;
def show : F<"show", "Show summary of the (indexed) codegen data file.">,
  Group<action_group>;
def convert : F<"convert", "Convert the (indexed) codegen data file in either text or binary format.">,
  Group<action_group>;
def merge : F<"merge", "Takes binary files having raw codegen data in custom sections, and merge them into an index codegen data file.">,
  Group<action_group>;
def cgdata_version : Flag<["-", "--"], "cgdata-version">, HelpText<"Display the cgdata version">;

// Output options
def output : Option<["-", "--"], "output", KIND_SEPARATE>,
             HelpText<"Create output file with specified name">, MetaVarName<"<file>">;
def o : JoinedOrSeparate<["-"], "o">, Alias<output>;
def format : Option<["-", "--"], "format", KIND_SEPARATE>,
             HelpText<"Specify the output format (text or binary)">, MetaVarName<"<value>">;
def f : JoinedOrSeparate<["-"], "f">, Alias<format>;
