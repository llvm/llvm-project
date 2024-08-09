include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

// General options
def generic_group : OptionGroup<"Genric Options">, HelpText<"Generic Options">;
def help : FF<"help", "Display this help">, Group<generic_group>;
def : F<"h", "Alias for --help">, Alias<help>, Group<generic_group>;
def version : FF<"version", "Display the LLVM version">, Group<generic_group>;
def : F<"v", "Alias for --version">, Alias<version>, Group<generic_group>;

// Action options
def action_group : OptionGroup<"Action">, HelpText<"Action (required)">;
def show : FF<"show", "Show summary of the (indexed) codegen data file.">,
  Group<action_group>;
def : F<"s", "Alias for --show">, Alias<show>, Group<action_group>;
def convert : FF<"convert", "Convert the (indexed) codegen data file in either text or binary format.">,
  Group<action_group>;
def : F<"c", "Alias for --convert">, Alias<convert>, Group<action_group>;
def merge : FF<"merge", "Take binary files having raw codegen data in custom sections, and merge them into an indexed codegen data file.">,
  Group<action_group>;
def : F<"m", "Alias for --merge">, Alias<merge>, Group<action_group>;

// Additional options
def cgdata_version : FF<"cgdata-version", "Display the cgdata version">;
def output : Option<["--"], "output", KIND_SEPARATE>,
             HelpText<"Specify the name for the output file to be created">, MetaVarName<"<file>">;
def : JoinedOrSeparate<["-"], "o">, Alias<output>, MetaVarName<"<file>">, HelpText<"Alias for --output">;
def format : Option<["--"], "format", KIND_SEPARATE>,
             HelpText<"Specify the output format (text or binary)">, MetaVarName<"<value>">;
def : JoinedOrSeparate<["-"], "f">, Alias<format>, HelpText<"Alias for --format">;
