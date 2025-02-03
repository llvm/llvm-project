include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--"], name #"=">, HelpText<help>;
  def : Separate<["--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def help : FF<"help", "Display this help">;
def : F<"h", "Alias for --help">, Alias<help>;
def version : FF<"version", "Display the version">;
def : F<"v", "Alias for --version">, Alias<version>;
def verbose : FF<"verbose", "Enable verbose logging and encoding details">;
defm convert :
  Eq<"convert",
     "Convert the specified file to the GSYM format.\nSupported files include ELF and mach-o files that will have their debug info (DWARF) and symbol table converted">;
def merged_functions :
  FF<"merged-functions", "When used with --convert, encodes merged function information for functions in debug info that have matching address ranges.\n"
                         "Without this option one function per unique address range will be emitted.\n"
                         "When used with --address/--addresses-from-stdin, all merged functions for a particular address will be displayed.\n"
                         "Without this option only one function will be displayed.">;
def dwarf_callsites : FF<"dwarf-callsites", "Load call site info from DWARF, if available">;
defm callsites_yaml_file :
  Eq<"callsites-yaml-file", "Load call site info from YAML file. Useful for testing.">, Flags<[HelpHidden]>;
defm arch :
  Eq<"arch",
     "Process debug information for the specified CPU architecture only.\nArchitectures may be specified by name or by number.\nThis option can be specified multiple times, once for each desired architecture">;
defm out_file :
  Eq<"out-file",
     "Specify the path where the converted GSYM file will be saved.\nWhen not specified, a '.gsym' extension will be appended to the file name specified in the --convert option">;
def : Separate<["-"], "o">, HelpText<"Alias for --out-file">, Alias<out_file_EQ>;
def verify : FF<"verify", "Verify the generated GSYM file against the information in the file that was converted">;
defm num_threads :
  Eq<"num-threads",
     "Specify the maximum number (n) of simultaneous threads to use when converting files to GSYM.\nDefaults to the number of cores on the current machine">;
defm segment_size :
  Eq<"segment-size",
     "Specify the size in bytes of the size the final GSYM file should be segmented into. This allows GSYM files to be split across multiple files">;
def quiet : FF<"quiet", "Do not output warnings about the debug information">;
defm address : Eq<"address", "Lookup an address in a GSYM file">;
def addresses_from_stdin :
  FF<"addresses-from-stdin",
     "Lookup addresses in a GSYM file that are read from stdin\nEach input line is expected to be of the following format: <addr> <gsym-path>">;
defm json_summary_file :
  Eq<"json-summary-file",
     "Output a categorized summary of errors into the JSON file specified.">;
defm merged_functions_filter : 
  Eq<"merged-functions-filter", 
     "When used with --address/--addresses-from-stdin and --merged-functions,\n"
     "filters the merged functions output to only show functions matching any of the specified regex patterns.\n"
     "Can be specified multiple times.">;
