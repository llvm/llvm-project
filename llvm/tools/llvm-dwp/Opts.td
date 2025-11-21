include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-", "--"], name>, HelpText<help>;
class S<string name, string help> : Separate<["-", "--"], name>, HelpText<help>;

def help : F<"help", "Display this help">;
def : F<"h", "Alias for --help">, Alias<help>;
def version : F<"version", "Display the version of this program">;

def execFileNames : S<"e", "Specify the executable/library files to get the list of *.dwo from.">, MetaVarName<"<filename>">;
def outputFileName : S<"o", "Specify the output file.">, MetaVarName<"<filename>">;
def continueOnCuIndexOverflow : Flag<["-", "--"], "continue-on-cu-index-overflow">;
def continueOnCuIndexOverflow_EQ : Joined<["-", "--"], "continue-on-cu-index-overflow=">,
  HelpText<"default = continue, This turns an error when offset \n"
    "\t\tfor .debug_*.dwo sections overfolws into a warning. = soft-stop, This produces a \n"
    "\t\ttruncated but valid DWP file, discarding any DWO files that would not fit within \n"
    "\t\tthe 32 bit/4GB limits of the format.">,
  Values<"continue,soft-stop">;

def dwarf64StringOffsets : Flag<["-", "--"], "dwarf64-str-offsets-promotion">;
def dwarf64StringOffsets_EQ
    : Joined<["-", "--"], "dwarf64-str-offsets-promotion=">,
      HelpText<"default = enabled, This allows .debug_str tables to exceed the "
               "4GB limit\n"
               "and have any DWARF32 .debug_str_offsets tables converted to "
               "DWARF64 only for tables\n"
               "that require 64 bit string offsets. = disabled, This setting "
               "doesn't convert DWARF32\n"
               ".debug_str_offsets tables in .dwo files to DWARF64 in the .dwp "
               "file. = always, This\n"
               "forces all .debug_str_offsets tables to be emitted as DWARF64. "
               "This is used for testing.">,
      Values<"disabled,enabled,always">;
