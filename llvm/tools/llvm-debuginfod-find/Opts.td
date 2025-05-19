include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-"], name>, HelpText<help>;
class FF<string name, string help>: Flag<["--"], name>, HelpText<help>;
class S<string name, string meta, string help>: Separate<["--"], name>, HelpText<help>, MetaVarName<meta>;

def help : FF<"help", "Display available options">;
def : F<"h", "Alias for --help">, Alias<help>;

def fetch_executable : FF<"executable", "If set, fetch a binary file associated with this build id, containing the executable sections.">;
def fetch_debuginfo : FF<"debuginfo", "If set, fetch a binary file associated with this build id, containing the debuginfo sections.">;
def fetch_source : S<"source", "<string>", "Fetch a source file associated with this build id, which is at this relative path relative to the compilation directory.">;
def dump_to_stdout : FF<"dump", "If set, dumps the contents of the fetched artifact "
                          "to standard output. Otherwise, dumps the absolute "
                          "path to the cached artifact on disk.">;
def debug_file_directory : S<"debug-file-directory", "<string>", "Path to directory where to look for debug files.">;

