include "llvm/Option/OptParser.td"

class F<string name, string help, list<SubCommand> subcommands = []> : Flag<["-", "--"], name, subcommands>, HelpText<help>;
class FF<string name, string help, list<SubCommand> subcommands = []> : Flag<["--"], name, subcommands>, HelpText<help>;

// Multiclass for value options that support both separate and joined format (with equals).
// E.g. --option value, --option=value, -option value, -option=value
multiclass Eq<string name, string help, list<SubCommand> subcommands = []> {
  def NAME : Separate<["-", "--"], name, subcommands>;
  def NAME #_eq : Joined<["-", "--"], name #"=", subcommands>,
                  Alias<!cast<Separate>(NAME)>,
                  HelpText<help>;
}

// Positional Subcommands
def show : SubCommand<"show", "Show summary or details of a profile.", "llvm-profdata show [options] <profdata-file>">;
def merge : SubCommand<"merge", "Merge multiple profiles together.", "llvm-profdata merge [options] <filename...>">;
def overlap : SubCommand<"overlap", "Compare and calculate the overlap of two profiles.", "llvm-profdata overlap [options] <base profile file> <test profile file>">;
def order : SubCommand<"order", "Generate function order list based on temporal profiling traces.", "llvm-profdata order [options] <profdata-file>">;

// Common Options
def help : Flag<["-", "--"], "help", [show, merge, overlap, order]>, HelpText<"Display this help">;
def : Flag<["-"], "h", [show, merge, overlap, order]>, Alias<help>;

def version : Flag<["-", "--"], "version", [show, merge, overlap, order]>, HelpText<"Display the version">;
def : Flag<["-"], "v", [show, merge, overlap, order]>, Alias<version>;

defm output : Eq<"output", "Output file", [show, merge, overlap, order]>;
def : JoinedOrSeparate<["-"], "o", [show, merge, overlap, order]>, Alias<output>;

// Show subcommand options
def counts : F<"counts", "Show counter values for shown functions", [show]>;
defm show_format : Eq<"show-format", "Emit output in the selected format if supported (text, json, yaml)", [show]>;
def text : F<"text", "Show instr profile data in text dump format (for show) or output text format (for merge)", [show, merge]>;
def json : F<"json", "Show sample profile data in the JSON format (deprecated, please use --show-format=json)", [show]>;
def ic_targets : F<"ic-targets", "Show indirect call site target values for shown functions", [show]>;
def show_vtables : F<"show-vtables", "Show vtable names for shown functions", [show]>;
def memop_sizes : F<"memop-sizes", "Show the profiled sizes of the memory intrinsic calls for shown functions", [show]>;
def detailed_summary : F<"detailed-summary", "Show detailed profile summary", [show]>;
defm detailed_summary_cutoffs : Eq<"detailed-summary-cutoffs", "Cutoff percentages (times 10000) for generating detailed summary", [show]>;
def hot_func_list : F<"hot-func-list", "Show profile summary of a list of hot functions", [show]>;
def all_functions : F<"all-functions", "Details for each and every function", [show]>;
def showcs : F<"showcs", "Show context sensitive counts", [show]>;
defm topn : Eq<"topn", "Show the list of functions with the largest internal counts", [show]>;
def list_below_cutoff : F<"list-below-cutoff", "Only output names of functions whose max count values are below the cutoff value", [show]>;
def show_prof_sym_list : F<"show-prof-sym-list", "Show profile symbol list if it exists in the profile.", [show]>;
def show_sec_info_only : F<"show-sec-info-only", "Show the information of each section in the sample profile. The flag is only usable when the sample profile is in extbinary format", [show]>;
def binary_ids : F<"binary-ids", "Show binary ids in the profile.", [show]>;
def temporal_profile_traces : F<"temporal-profile-traces", "Show temporal profile traces in the profile.", [show]>;
def covered : F<"covered", "Show only the functions that have been executed.", [show]>;
def profile_version : F<"profile-version", "Show profile version.", [show]>;

// Overlap subcommand options
def cs : F<"cs", "For context sensitive PGO counts. Does not work with CSSPGO.", [overlap]>;
defm similarity_cutoff : Eq<"similarity-cutoff", "For sample profiles, list function names (with calling context for csspgo) for overlapped functions with similarities below the cutoff (percentage times 10000).", [overlap]>;

// Order subcommand options
defm num_test_traces : Eq<"num-test-traces", "Keep aside the last <num-test-traces> traces in the profile when computing the function order and instead use them to evaluate that order", [order]>;

// Profile Kind standalone options (legacy cl compatibility)
def sample : Flag<["-", "--"], "sample", [show, merge, overlap]>, HelpText<"Sample profile">;
def instr : Flag<["-", "--"], "instr", [show, merge, overlap]>, HelpText<"Instrumentation profile">;
def memory : Flag<["-", "--"], "memory", [show]>, HelpText<"MemProf memory access profile">;

// Multi-subcommand Shared Options
defm value_cutoff : Eq<"value-cutoff", "Set the count value cutoff. Functions with the maximum count less than this value will not be printed out. (Default is 0)", [show, overlap]>;
defm max_debug_info_correlation_warnings : Eq<"max-debug-info-correlation-warnings", "The maximum number of warnings to emit when correlating profile from debug info (0 = no limit)", [show, merge]>;
defm profiled_binary : Eq<"profiled-binary", "Path to binary from which the profile was collected.", [show, merge]>;
defm debug_info : Eq<"debug-info", "Extract profile metadata from debug info (for show) or correlate raw profile (for merge).", [show, merge]>;
defm debug_file_directory : Eq<"debug-file-directory", "Directories to search for object files by build ID", [show, merge]>;
defm correlate : Eq<"correlate", "Use debug-info or binary correlation to correlate profiles with build id fetcher", [show, merge]>;
defm function : Eq<"function", "Only functions matching the filter are shown in the output.", [show, merge, overlap]>;
defm fs_discriminator_pass : Eq<"fs-discriminator-pass", "Zero out the discriminator bits for the FS discriminator pass beyond this value: base, pass1, pass2, pass3, passlast", [show, merge]>;

// Merge subcommand options
defm binary_file : Eq<"binary-file", "Use the provided unstripped binary to correlate the raw profile.", [merge]>;
def debuginfod : F<"debuginfod", "Enable debuginfod", [merge]>;
defm weighted_input : Eq<"weighted-input", "Input file with weight specified as <weight>,<filename>", [merge]>;
def binary : Flag<["-", "--"], "binary", [merge]>, HelpText<"Output binary format">;
def extbinary : Flag<["-", "--"], "extbinary", [merge]>, HelpText<"Output extbinary format">;
def gcc : Flag<["-", "--"], "gcc", [merge]>, HelpText<"Output gcc format (only meaningful for -sample)">;
defm input_files : Eq<"input-files", "Path to file containing newline-separated [<weight>,]<filename> entries", [merge]>;
def : JoinedOrSeparate<["-"], "f", [merge]>, Alias<input_files>;
def dump_input_file_list : F<"dump-input-file-list", "Dump the list of input files and their weights, then exit", [merge]>;
defm remapping_file : Eq<"remapping-file", "Symbol remapping file", [merge]>;
def : JoinedOrSeparate<["-"], "r", [merge]>, Alias<remapping_file>;
def use_md5 : F<"use-md5", "Choose to use MD5 to represent string in name table (only meaningful for -extbinary)", [merge]>;
def compress_all_sections : F<"compress-all-sections", "Compress all sections when writing the profile (only meaningful for -extbinary)", [merge]>;
def sample_merge_cold_context : F<"sample-merge-cold-context", "Merge context sample profiles whose count is below cold threshold", [merge]>;
def sample_trim_cold_context : F<"sample-trim-cold-context", "Trim context sample profiles whose count is below cold threshold", [merge]>;
defm sample_frame_depth_for_cold_context : Eq<"sample-frame-depth-for-cold-context", "Keep the last K frames while merging cold profile. 1 means the context-less base profile", [merge]>;
defm output_size_limit : Eq<"output-size-limit", "Trim cold functions until profile size is below specified limit in bytes. This uses a heuristic and functions may be excessively trimmed", [merge]>;
def gen_partial_profile : F<"gen-partial-profile", "Generate a partial profile (only meaningful for -extbinary)", [merge]>;
def split_layout : F<"split-layout", "Split the profile to two sections with one containing sample profiles with inlined functions and the other without (only meaningful for -extbinary)", [merge]>;
defm supplement_instr_with_sample : Eq<"supplement-instr-with-sample", "Supplement an instr profile with sample profile, to correct the profile unrepresentativeness issue. The sample profile is the input of the flag. Output will be in instr format (The flag only works with -instr)", [merge]>;
defm zero_counter_threshold : Eq<"zero-counter-threshold", "For the function which is cold in instr profile but hot in sample profile, if the ratio of the number of zero counters divided by the total number of counters is above the threshold, the profile of the function will be regarded as being harmful for performance and will be dropped.", [merge]>;
defm suppl_min_size_threshold : Eq<"suppl-min-size-threshold", "If the size of a function is smaller than the threshold, assume it can be inlined by PGO early inliner and it won't be adjusted based on sample profile.", [merge]>;
defm instr_prof_cold_threshold : Eq<"instr-prof-cold-threshold", "User specified cold threshold for instr profile which will override the cold threshold got from profile summary. ", [merge]>;
defm temporal_profile_trace_reservoir_size : Eq<"temporal-profile-trace-reservoir-size", "The maximum number of stored temporal profile traces (default: 100)", [merge]>;
defm temporal_profile_max_trace_length : Eq<"temporal-profile-max-trace-length", "The maximum length of a single temporal profile trace (default: 10000)", [merge]>;
defm no_function : Eq<"no-function", "Exclude functions matching the filter from the output.", [merge]>;
defm failure_mode : Eq<"failure-mode", "Failure mode: warn (Do not fail and just print warnings.), any (Fail if any profile is invalid.), all (Fail only if all profiles are invalid.)", [merge]>;
def sparse : F<"sparse", "Generate a sparse profile (only meaningful for -instr)", [merge]>;
defm num_threads : Eq<"num-threads", "Number of merge threads to use (default: autodetect)", [merge]>;
def : JoinedOrSeparate<["-"], "j", [merge]>, Alias<num_threads>;
defm prof_sym_list : Eq<"prof-sym-list", "Path to file containing the list of function symbols used to populate profile symbol list", [merge]>;
defm convert_sample_profile_layout : Eq<"convert-sample-profile-layout", "Convert the generated profile to a profile with a new layout: nest, flat", [merge]>;
def drop_profile_symbol_list : F<"drop-profile-symbol-list", "Drop the profile symbol list when merging AutoFDO profiles (only meaningful for -sample)", [merge]>;
def keep_vtable_symbols : F<"keep-vtable-symbols", "If true, keep the vtable symbols in indexed profiles", [merge]>;
def write_prev_version : F<"write-prev-version", "Write the previous version of indexed format, to enable some forward compatibility.", [merge]>;
defm memprof_version : Eq<"memprof-version", "Specify the version of the memprof format to use: 2, 3, 4", [merge]>;
def memprof_full_schema : F<"memprof-full-schema", "Use the full schema for serialization", [merge]>;
def memprof_random_hotness : F<"memprof-random-hotness", "Generate random hotness values", [merge]>;
defm memprof_random_hotness_seed : Eq<"memprof-random-hotness-seed", "Random hotness seed to use (0 to generate new seed)", [merge]>;
