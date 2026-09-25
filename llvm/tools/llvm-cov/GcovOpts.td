include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--"], name #"=">, HelpText<help>;
  def : Separate<["--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def all_blocks : FF<"all-blocks", "Display all basic blocks">;
def branch_counts : FF<"branch-counts", "Display branch counts instead of percentages (requires -b)">;
def branch_probabilities : FF<"branch-probabilities", "Display branch probabilities">;
def demangled_names : FF<"demangled-names", "Demangle function names">;
def dump_gcov : FF<"dump", "Dump the gcov file to stderr">;
def function_summaries : FF<"function-summaries", "Show coverage for each function">;
defm gcda : Eq<"gcda", "Override inferred gcda file">, MetaVarName<"<file>">;
defm gcno : Eq<"gcno", "Override inferred gcno file">, MetaVarName<"<file>">;
def hash_filenames : FF<"hash-filenames", "Hash long pathnames">;
def help : FF<"help", "Display this help">;
// Supported by gcov 4.9~8. gcov 9 (GCC r265587) removed --intermediate-format
// and -i was changed to mean --json-format. We consider this format still
// useful and support -i.
def intermediate_format : FF<"intermediate-format", "Output .gcov in intermediate text format">;
def long_file_names : FF<"long-file-names", "Prefix filenames with the main file">;
def no_output : FF<"no-output", "Do not output any .gcov files">;
defm object_directory : Eq<"object-directory", "Find objects in DIR or based on FILE's path">, MetaVarName<"<DIR>">;
def : Joined<["--"], "object-file=">, Alias<object_directory_EQ>, HelpText<"Alias for --object-directory">, MetaVarName<"<FILE>">;
def : Separate<["--"], "object-file">, Alias<object_directory_EQ>;
def preserve_paths : FF<"preserve-paths", "Preserve path components">;
def relative_only : FF<"relative-only", "Only dump files with relative paths or absolute paths with the prefix specified by -s">;
defm source_prefix : Eq<"source-prefix", "Source prefix to elide">, MetaVarName<"<prefix>">;
def print_stdout : FF<"stdout", "Print to stdout">;
def unconditional_branches : FF<"unconditional-branches", "Display unconditional branch info (requires -b)">;
def version : FF<"version", "Display the version">;

def : F<"a", "Alias for --all-blocks">, Alias<all_blocks>;
def : F<"b", "Alias for --branch-probabilities">, Alias<branch_probabilities>;
def : F<"c", "Alias for --branch-counts">, Alias<branch_counts>;
def : F<"f", "Alias for --function-summaries">, Alias<function_summaries>;
def : F<"i", "Alias for --intermediate-format">, Alias<intermediate_format>;
def : F<"l", "Alias for --long-file-names">, Alias<long_file_names>;
def : F<"m", "Alias for --demangled-names">, Alias<demangled_names>;
def : F<"n", "Alias for --no-output">, Alias<no_output>;
def : JoinedOrSeparate<["-"], "o">, Alias<object_directory_EQ>, HelpText<"Alias for --object-directory">, MetaVarName<"<DIR|FILE>">;
def : F<"p", "Alias for --preserve-paths">, Alias<preserve_paths>;
def : F<"r", "Alias for --relative-only">, Alias<relative_only>;
def : JoinedOrSeparate<["-"], "s">, Alias<source_prefix_EQ>, HelpText<"Alias for --source-prefix">, MetaVarName<"<prefix>">;
def : F<"t", "Alias for --stdout">, Alias<print_stdout>;
def : F<"u", "Alias for --unconditional-branches">, Alias<unconditional_branches>;
def : F<"x", "Alias for --hash-filenames">, Alias<hash_filenames>;
