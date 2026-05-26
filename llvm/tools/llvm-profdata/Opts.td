include "llvm/Option/OptParser.td"

class F<string name, string help, list<SubCommand> subcommands = []> : Flag<["-", "--"], name, subcommands>, HelpText<help>;
class FF<string name, string help, list<SubCommand> subcommands = []> : Flag<["--"], name, subcommands>, HelpText<help>;

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

def output : JoinedOrSeparate<["-", "--"], "output", [show, merge, overlap, order]>, HelpText<"Output file">, MetaVarName<"<file>">;
def : JoinedOrSeparate<["-"], "o", [show, merge, overlap, order]>, Alias<output>;

// Show subcommand options
def counts : F<"counts", "Show counter values for shown functions", [show]>;
def show_format : JoinedOrSeparate<["-", "--"], "show-format", [show]>, HelpText<"Emit output in the selected format if supported (text, json, yaml)">, MetaVarName<"<format>">;
def text : F<"text", "Show instr profile data in text dump format", [show]>;
def json : F<"json", "Show sample profile data in the JSON format (deprecated, please use --show-format=json)", [show]>;
def ic_targets : F<"ic-targets", "Show indirect call site target values for shown functions", [show]>;
def show_vtables : F<"show-vtables", "Show vtable names for shown functions", [show]>;
def memop_sizes : F<"memop-sizes", "Show the profiled sizes of the memory intrinsic calls for shown functions", [show]>;
def detailed_summary : F<"detailed-summary", "Show detailed profile summary", [show]>;
def detailed_summary_cutoffs : JoinedOrSeparate<["-", "--"], "detailed-summary-cutoffs", [show]>, HelpText<"Cutoff percentages (times 10000) for generating detailed summary">, MetaVarName<"<cutoffs>">;
def hot_func_list : F<"hot-func-list", "Show profile summary of a list of hot functions", [show]>;
def all_functions : F<"all-functions", "Details for each and every function", [show]>;
def showcs : F<"showcs", "Show context sensitive counts", [show]>;
def topn : JoinedOrSeparate<["-", "--"], "topn", [show]>, HelpText<"Show the list of functions with the largest internal counts">, MetaVarName<"<n>">;
def list_below_cutoff : F<"list-below-cutoff", "Only output names of functions whose max count values are below the cutoff value", [show]>;
def show_prof_sym_list : F<"show-prof-sym-list", "Show profile symbol list if it exists in the profile.", [show]>;
def show_sec_info_only : F<"show-sec-info-only", "Show the information of each section in the sample profile. The flag is only usable when the sample profile is in extbinary format", [show]>;
def binary_ids : F<"binary-ids", "Show binary ids in the profile.", [show]>;
def temporal_profile_traces : F<"temporal-profile-traces", "Show temporal profile traces in the profile.", [show]>;
def covered : F<"covered", "Show only the functions that have been executed.", [show]>;
def profile_version : F<"profile-version", "Show profile version.", [show]>;

// Overlap subcommand options
def cs : F<"cs", "For context sensitive PGO counts. Does not work with CSSPGO.", [overlap]>;
def similarity_cutoff : JoinedOrSeparate<["-", "--"], "similarity-cutoff", [overlap]>, HelpText<"For sample profiles, list function names (with calling context for csspgo) for overlapped functions with similarities below the cutoff (percentage times 10000).">, MetaVarName<"<cutoff>">;


// Order subcommand options
def num_test_traces : JoinedOrSeparate<["-", "--"], "num-test-traces", [order]>, HelpText<"Keep aside the last <num-test-traces> traces in the profile when computing the function order and instead use them to evaluate that order">, MetaVarName<"<traces>">;

// Multi-subcommand Shared Options
def profile_kind : JoinedOrSeparate<["-", "--"], "profile-kind", [show, merge, overlap]>, HelpText<"Profile kind: instr, sample (show also supports memory)">, MetaVarName<"<kind>">;
def value_cutoff : JoinedOrSeparate<["-", "--"], "value-cutoff", [show, overlap]>, HelpText<"Set the count value cutoff. Functions with the maximum count less than this value will not be printed out. (Default is 0)">, MetaVarName<"<cutoff>">;
def max_debug_info_correlation_warnings : JoinedOrSeparate<["-", "--"], "max-debug-info-correlation-warnings", [show, merge]>, HelpText<"The maximum number of warnings to emit when correlating profile from debug info (0 = no limit)">, MetaVarName<"<warnings>">;
def profiled_binary : JoinedOrSeparate<["-", "--"], "profiled-binary", [show, merge]>, HelpText<"Path to binary from which the profile was collected.">, MetaVarName<"<binary>">;
def debug_info : JoinedOrSeparate<["-", "--"], "debug-info", [show, merge]>, HelpText<"Extract profile metadata from debug info (for show) or correlate raw profile (for merge).">, MetaVarName<"<file>">;
def debug_file_directory : JoinedOrSeparate<["-", "--"], "debug-file-directory", [show, merge]>, HelpText<"Directories to search for object files by build ID">, MetaVarName<"<dir>">;
def correlate : JoinedOrSeparate<["-", "--"], "correlate", [show, merge]>, HelpText<"Use debug-info or binary correlation to correlate profiles with build id fetcher">, MetaVarName<"<kind>">;
def function : JoinedOrSeparate<["-", "--"], "function", [show, merge, overlap]>, HelpText<"Only functions matching the filter are shown in the output.">, MetaVarName<"<filter>">;
def fs_discriminator_pass : JoinedOrSeparate<["-", "--"], "fs-discriminator-pass", [show, merge]>, HelpText<"Zero out the discriminator bits for the FS discriminator pass beyond this value: base, pass1, pass2, pass3, passlast">, MetaVarName<"<pass>">;

// Merge subcommand options
def binary_file : JoinedOrSeparate<["-", "--"], "binary-file", [merge]>, HelpText<"Use the provided unstripped binary to correlate the raw profile.">, MetaVarName<"<file>">;
def debuginfod : F<"debuginfod", "Enable debuginfod", [merge]>;
def weighted_input : JoinedOrSeparate<["-", "--"], "weighted-input", [merge]>, HelpText<"Input file with weight specified as <weight>,<filename>">, MetaVarName<"<weight,filename>">;
def format : JoinedOrSeparate<["-", "--"], "format", [merge]>, HelpText<"Format of output profile: binary, extbinary (default), text, gcc (only for -sample)">, MetaVarName<"<format>">;
def input_files : JoinedOrSeparate<["-", "--"], "input-files", [merge]>, HelpText<"Path to file containing newline-separated [<weight>,]<filename> entries">, MetaVarName<"<file>">;
def : JoinedOrSeparate<["-"], "f", [merge]>, Alias<input_files>;
def dump_input_file_list : F<"dump-input-file-list", "Dump the list of input files and their weights, then exit", [merge]>;
def remapping_file : JoinedOrSeparate<["-", "--"], "remapping-file", [merge]>, HelpText<"Symbol remapping file">, MetaVarName<"<file>">;
def : JoinedOrSeparate<["-"], "r", [merge]>, Alias<remapping_file>;
def use_md5 : F<"use-md5", "Choose to use MD5 to represent string in name table (only meaningful for -extbinary)", [merge]>;
def compress_all_sections : F<"compress-all-sections", "Compress all sections when writing the profile (only meaningful for -extbinary)", [merge]>;
def sample_merge_cold_context : F<"sample-merge-cold-context", "Merge context sample profiles whose count is below cold threshold", [merge]>;
def sample_trim_cold_context : F<"sample-trim-cold-context", "Trim context sample profiles whose count is below cold threshold", [merge]>;
def sample_frame_depth_for_cold_context : JoinedOrSeparate<["-", "--"], "sample-frame-depth-for-cold-context", [merge]>, HelpText<"Keep the last K frames while merging cold profile. 1 means the context-less base profile">, MetaVarName<"<depth>">;
def output_size_limit : JoinedOrSeparate<["-", "--"], "output-size-limit", [merge]>, HelpText<"Trim cold functions until profile size is below specified limit in bytes. This uses a heuristic and functions may be excessively trimmed">, MetaVarName<"<limit>">;
def gen_partial_profile : F<"gen-partial-profile", "Generate a partial profile (only meaningful for -extbinary)", [merge]>;
def split_layout : F<"split-layout", "Split the profile to two sections with one containing sample profiles with inlined functions and the other without (only meaningful for -extbinary)", [merge]>;
def supplement_instr_with_sample : JoinedOrSeparate<["-", "--"], "supplement-instr-with-sample", [merge]>, HelpText<"Supplement an instr profile with sample profile, to correct the profile unrepresentativeness issue. The sample profile is the input of the flag. Output will be in instr format (The flag only works with -instr)">, MetaVarName<"<file>">;
def zero_counter_threshold : JoinedOrSeparate<["-", "--"], "zero-counter-threshold", [merge]>, HelpText<"For the function which is cold in instr profile but hot in sample profile, if the ratio of the number of zero counters divided by the total number of counters is above the threshold, the profile of the function will be regarded as being harmful for performance and will be dropped.">, MetaVarName<"<threshold>">;
def suppl_min_size_threshold : JoinedOrSeparate<["-", "--"], "suppl-min-size-threshold", [merge]>, HelpText<"If the size of a function is smaller than the threshold, assume it can be inlined by PGO early inliner and it won't be adjusted based on sample profile.">, MetaVarName<"<threshold>">;
def instr_prof_cold_threshold : JoinedOrSeparate<["-", "--"], "instr-prof-cold-threshold", [merge]>, HelpText<"User specified cold threshold for instr profile which will override the cold threshold got from profile summary. ">, MetaVarName<"<threshold>">;
def temporal_profile_trace_reservoir_size : JoinedOrSeparate<["-", "--"], "temporal-profile-trace-reservoir-size", [merge]>, HelpText<"The maximum number of stored temporal profile traces (default: 100)">, MetaVarName<"<size>">;
def temporal_profile_max_trace_length : JoinedOrSeparate<["-", "--"], "temporal-profile-max-trace-length", [merge]>, HelpText<"The maximum length of a single temporal profile trace (default: 10000)">, MetaVarName<"<length>">;
def no_function : JoinedOrSeparate<["-", "--"], "no-function", [merge]>, HelpText<"Exclude functions matching the filter from the output.">, MetaVarName<"<filter>">;
def failure_mode : JoinedOrSeparate<["-", "--"], "failure-mode", [merge]>, HelpText<"Failure mode: warn (Do not fail and just print warnings.), any (Fail if any profile is invalid.), all (Fail only if all profiles are invalid.)">, MetaVarName<"<mode>">;
def sparse : F<"sparse", "Generate a sparse profile (only meaningful for -instr)", [merge]>;
def num_threads : JoinedOrSeparate<["-", "--"], "num-threads", [merge]>, HelpText<"Number of merge threads to use (default: autodetect)">, MetaVarName<"<threads>">;
def : JoinedOrSeparate<["-"], "j", [merge]>, Alias<num_threads>;
def prof_sym_list : JoinedOrSeparate<["-", "--"], "prof-sym-list", [merge]>, HelpText<"Path to file containing the list of function symbols used to populate profile symbol list">, MetaVarName<"<file>">;
def convert_sample_profile_layout : JoinedOrSeparate<["-", "--"], "convert-sample-profile-layout", [merge]>, HelpText<"Convert the generated profile to a profile with a new layout: nest, flat">, MetaVarName<"<layout>">;
def drop_profile_symbol_list : F<"drop-profile-symbol-list", "Drop the profile symbol list when merging AutoFDO profiles (only meaningful for -sample)", [merge]>;
def keep_vtable_symbols : F<"keep-vtable-symbols", "If true, keep the vtable symbols in indexed profiles", [merge]>;
def write_prev_version : F<"write-prev-version", "Write the previous version of indexed format, to enable some forward compatibility.", [merge]>;
def memprof_version : JoinedOrSeparate<["-", "--"], "memprof-version", [merge]>, HelpText<"Specify the version of the memprof format to use: 2, 3, 4">, MetaVarName<"<version>">;
def memprof_full_schema : F<"memprof-full-schema", "Use the full schema for serialization", [merge]>;
def memprof_random_hotness : F<"memprof-random-hotness", "Generate random hotness values", [merge]>;
def memprof_random_hotness_seed : JoinedOrSeparate<["-", "--"], "memprof-random-hotness-seed", [merge]>, HelpText<"Random hotness seed to use (0 to generate new seed)">, MetaVarName<"<seed>">;
