include "llvm/Option/OptParser.td"

def Show : SubCommand<"show",
                      "Takes a profile data file and displays the profiles. "
                      "See detailed documentation in "
                      "https://llvm.org/docs/CommandGuide/llvm-profdata.html"
                      "#profdata-show">;
def Order : SubCommand<"order",
                       "Reads temporal profiling traces from a profile and "
                       "outputs a function order that reduces page faults. "
                       "See detailed documentation in "
                       "https://llvm.org/docs/CommandGuide/llvm-profdata.html"
                       "#profdata-order">;
def Overlap : SubCommand<"overlap",
                         "Computes the overlap between two profiles. See "
                         "detailed documentation in "
                         "https://llvm.org/docs/CommandGuide/llvm-profdata.html"
                         "#profdata-overlap">;
def Merge : SubCommand<"merge",
                       "Takes several profiles and merge them together. See "
                       "detailed documentation in "
                       "https://llvm.org/docs/CommandGuide/llvm-profdata.html"
                       "#profdata-merge">;

def help : Flag<["--"], "help">, HelpText<"Display this help">;
def version : Flag<["--"], "version">, HelpText<"Display the version">;

def output : JoinedOrSeparate<["--"], "output",
                               [Show, Order, Overlap, Merge]>,
             MetaVarName<"<output>">,
             HelpText<"Output file">;
def : JoinedOrSeparate<["-"], "o", [Show, Order, Overlap, Merge]>,
     Alias<output>, HelpText<"Alias for --output">;

def instr : Flag<["--"], "instr", [Show, Overlap, Merge]>,
            HelpText<"Instrumentation profile (default)">;
def sample : Flag<["--"], "sample", [Show, Overlap, Merge]>,
             HelpText<"Sample profile">;
def memory : Flag<["--"], "memory", [Show]>,
             HelpText<"MemProf memory access profile">;

def max_debug_info_correlation_warnings
    : JoinedOrSeparate<["--"], "max-debug-info-correlation-warnings",
                       [Show, Merge]>,
      HelpText<"Maximum number of warnings to emit when correlating profile "
               "from debug info (0 = no limit)">,
      MetaVarName<"<n>">;
def profiled_binary
    : JoinedOrSeparate<["--"], "profiled-binary", [Show, Merge]>,
      HelpText<"Path to binary from which the profile was collected.">,
      MetaVarName<"<binary>">;
def debug_info
    : JoinedOrSeparate<["--"], "debug-info", [Show, Merge]>,
      HelpText<"For show, read and extract profile metadata from debug info. "
               "For merge, correlate the raw profile using the provided "
               "debug info.">,
      MetaVarName<"<file>">;
def binary_file
    : JoinedOrSeparate<["--"], "binary-file", [Merge]>,
      HelpText<"Use the provided unstripped binary to correlate the raw "
               "profile.">,
      MetaVarName<"<file>">;
def debug_file_directory
    : JoinedOrSeparate<["--"], "debug-file-directory",
                       [Show, Order, Overlap, Merge]>,
      HelpText<"Directories to search for object files by build ID">,
      MetaVarName<"<dir>">;
def debuginfod : Flag<["--"], "debuginfod", [Merge]>,
                 HelpText<"Enable debuginfod">;
def correlate : JoinedOrSeparate<["--"], "correlate",
                                  [Show, Order, Overlap, Merge]>,
                HelpText<"Use debug-info or binary correlation to correlate "
                         "profiles with build id fetcher">,
                MetaVarName<"<mode>">;
def function : JoinedOrSeparate<["--"], "function",
                                [Show, Overlap, Merge]>,
               HelpText<"Only functions matching the filter are shown or "
                        "merged.">,
               MetaVarName<"<regex>">;

def weighted_input
    : JoinedOrSeparate<["--"], "weighted-input", [Merge]>,
      HelpText<"<weight>,<filename>">, MetaVarName<"<weight>,<file>">;
def binary : Flag<["--"], "binary", [Merge]>,
             HelpText<"Binary encoding">;
def extbinary : Flag<["--"], "extbinary", [Merge]>,
                HelpText<"Extensible binary encoding (default)">;
def text : Flag<["--"], "text", [Show, Merge]>,
           HelpText<"Text output format">;
def gcc : Flag<["--"], "gcc", [Merge]>,
          HelpText<"GCC encoding (only meaningful for -sample)">;
def input_files
    : JoinedOrSeparate<["--"], "input-files", [Merge]>,
      HelpText<"Path to file containing newline-separated "
               "[<weight>,]<filename> entries">,
      MetaVarName<"<file>">;
def : JoinedOrSeparate<["-"], "f", [Merge]>, Alias<input_files>,
     HelpText<"Alias for --input-files">;
def dump_input_file_list : Flag<["--"], "dump-input-file-list", [Merge]>,
                           HelpText<"Dump the list of input files and their "
                                    "weights, then exit">;

def remapping_file
    : JoinedOrSeparate<["--"], "remapping-file", [Merge]>,
      HelpText<"Symbol remapping file">, MetaVarName<"<file>">;
def : JoinedOrSeparate<["-"], "r", [Merge]>, Alias<remapping_file>,
     HelpText<"Alias for --remapping-file">;
def use_md5 : Flag<["--"], "use-md5", [Merge]>,
              HelpText<"Use MD5 to represent strings in the name table "
                       "(only meaningful for -extbinary)">;

def compress_all_sections
    : Flag<["--"], "compress-all-sections", [Merge]>,
      HelpText<"Compress all sections when writing the profile (only "
               "meaningful for -extbinary)">;

def sample_merge_cold_context
    : Flag<["--"], "sample-merge-cold-context", [Merge]>,
      HelpText<"Merge context sample profiles whose count is below cold "
               "threshold">;

def sample_trim_cold_context
    : Flag<["--"], "sample-trim-cold-context", [Merge]>,
      HelpText<"Trim context sample profiles whose count is below cold "
               "threshold">;

def sample_frame_depth_for_cold_context
    : JoinedOrSeparate<["--"], "sample-frame-depth-for-cold-context",
                       [Merge]>,
      HelpText<"Keep the last K frames while merging cold profile. 1 means "
               "the context-less base profile">,
      MetaVarName<"<depth>">;
def output_size_limit
    : JoinedOrSeparate<["--"], "output-size-limit", [Merge]>,
      HelpText<"Trim cold functions until profile size is below specified "
               "limit in bytes">,
      MetaVarName<"<bytes>">;
def gen_partial_profile
    : Flag<["--"], "gen-partial-profile", [Merge]>,
      HelpText<"Generate a partial profile (only meaningful for -extbinary)">;

def split_layout : Flag<["--"], "split-layout", [Merge]>,
                   HelpText<"Split the profile into sections with and without "
                            "inlined functions (only meaningful for "
                            "-extbinary)">;

def supplement_instr_with_sample
    : JoinedOrSeparate<["--"], "supplement-instr-with-sample", [Merge]>,
      HelpText<"Supplement an instr profile with a sample profile. Output "
               "will be in instr format.">,
      MetaVarName<"<sample-profile>">;
def zero_counter_threshold
    : JoinedOrSeparate<["--"], "zero-counter-threshold", [Merge]>,
      HelpText<"Ratio of zero counters required to drop a function when "
               "supplementing instr profiles">,
      MetaVarName<"<ratio>">;
def suppl_min_size_threshold
    : JoinedOrSeparate<["--"], "suppl-min-size-threshold", [Merge]>,
      HelpText<"Assume functions smaller than this threshold can be inlined "
               "and will not be adjusted based on sample profile.">,
      MetaVarName<"<n>">;
def instr_prof_cold_threshold
    : JoinedOrSeparate<["--"], "instr-prof-cold-threshold", [Merge]>,
      HelpText<"User specified cold threshold for instr profile to override "
               "the cold threshold from profile summary.">,
      MetaVarName<"<n>">;
def temporal_profile_trace_reservoir_size
    : JoinedOrSeparate<["--"], "temporal-profile-trace-reservoir-size",
                       [Merge]>,
      HelpText<"Maximum number of stored temporal profile traces (default: "
               "100)">,
      MetaVarName<"<n>">;
def temporal_profile_max_trace_length
    : JoinedOrSeparate<["--"], "temporal-profile-max-trace-length",
                       [Merge]>,
      HelpText<"Maximum length of a single temporal profile trace "
               "(default: 10000)">,
      MetaVarName<"<n>">;
def no_function : JoinedOrSeparate<["--"], "no-function", [Merge]>,
                  HelpText<"Exclude functions matching the filter from the "
                           "output.">,
                  MetaVarName<"<regex>">;
def failure_mode
    : JoinedOrSeparate<["--"], "failure-mode", [Merge]>,
      HelpText<"Failure mode: warn, any, or all">,
      MetaVarName<"<mode>">;
def sparse
    : Flag<["--"], "sparse", [Merge]>,
      HelpText<"Generate a sparse profile (only meaningful for -instr)">;

def profile_isfs : Flag<["--"], "profile-isfs", [Show, Merge]>,
                   HelpText<"Profile uses flow-sensitive discriminators">;


def generate_merged_base_profiles
    : Flag<["--"], "generate-merged-base-profiles", [Merge]>,
      HelpText<"Generate merged base profiles when creating nested "
               "context-sensitive profiles">;


def profile_symbol_list_cutoff
    : JoinedOrSeparate<["--"], "profile-symbol-list-cutoff", [Merge]>,
      HelpText<"Maximum number of symbols to keep in the profile symbol list">;

def profile_summary_cutoff_hot
    : JoinedOrSeparate<["--"], "profile-summary-cutoff-hot",
                       [Show, Merge]>,
      HelpText<"Percentile (times 10000) for determining hot threshold">;

def profile_summary_cutoff_cold
    : JoinedOrSeparate<["--"], "profile-summary-cutoff-cold",
                       [Show, Merge]>,
      HelpText<"Percentile (times 10000) for determining cold threshold">;

def profile_summary_hot_count
    : JoinedOrSeparate<["--"], "profile-summary-hot-count",
                       [Show, Merge]>,
      HelpText<"Override hot count derived from summary cutoffs">;

def profile_summary_cold_count
    : JoinedOrSeparate<["--"], "profile-summary-cold-count",
                       [Show, Merge]>,
      HelpText<"Override cold count derived from summary cutoffs">;

def profile_summary_contextless
    : JoinedOrSeparate<["--"], "profile-summary-contextless",
                       [Show, Merge]>,
      HelpText<"Merge context profiles before calculating summary thresholds">;

def profile_summary_huge_working_set_size_threshold
    : JoinedOrSeparate<["--"],
                       "profile-summary-huge-working-set-size-threshold",
                       [Show, Merge]>,
      HelpText<"Threshold for huge working set size used in profile summary">;

def profile_summary_large_working_set_size_threshold
    : JoinedOrSeparate<["--"],
                       "profile-summary-large-working-set-size-threshold",
                       [Show, Merge]>,
      HelpText<"Threshold for large working set size used in profile summary">;

def num_threads : JoinedOrSeparate<["--"], "num-threads", [Merge]>,
                  HelpText<"Number of merge threads to use (default: autodetect)">,
                  MetaVarName<"<n>">;
def : JoinedOrSeparate<["-"], "j", [Merge]>, Alias<num_threads>,
     HelpText<"Alias for --num-threads">;
def prof_sym_list
    : JoinedOrSeparate<["--"], "prof-sym-list", [Merge]>,
      HelpText<"Path to file containing the list of function symbols used to "
               "populate profile symbol list">,
      MetaVarName<"<file>">;
def convert_sample_profile_layout
    : JoinedOrSeparate<["--"], "convert-sample-profile-layout", [Merge]>,
      HelpText<"Convert the generated profile to a new layout: nest or flat">,
      MetaVarName<"<layout>">;
def drop_profile_symbol_list
    : Flag<["--"], "drop-profile-symbol-list", [Merge]>,
      HelpText<"Drop the profile symbol list when merging AutoFDO profiles "
               "(only meaningful for -sample)">;

def keep_vtable_symbols
    : Flag<["--"], "keep-vtable-symbols", [Merge]>,
      HelpText<"Keep the vtable symbols in indexed profiles">;

def write_prev_version
    : Flag<["--"], "write-prev-version", [Merge]>,
      HelpText<"Write the previous version of indexed format for forward "
               "compatibility.">;

def memprof_version
    : JoinedOrSeparate<["--"], "memprof-version", [Merge]>,
      HelpText<"Specify the version of the memprof format to use (2, 3, or 4)">,
      MetaVarName<"<n>">;
def memprof_full_schema
    : Flag<["--"], "memprof-full-schema", [Merge]>,
      HelpText<"Use the full schema for serialization">;

def memprof_random_hotness
    : Flag<["--"], "memprof-random-hotness", [Merge]>,
      HelpText<"Generate random hotness values">;

def memprof_random_hotness_seed
    : JoinedOrSeparate<["--"], "memprof-random-hotness-seed", [Merge]>,
      HelpText<"Random hotness seed to use (0 to generate new seed)">,
      MetaVarName<"<n>">;

def similarity_cutoff
    : JoinedOrSeparate<["--"], "similarity-cutoff", [Overlap]>,
      HelpText<"List overlapped functions with similarities below the cutoff "
               "(percentage times 10000).">,
      MetaVarName<"<n>">;
def cs : Flag<["--"], "cs", [Overlap]>,
         HelpText<"For context sensitive PGO counts. Does not work with "
                  "CSSPGO.">;
def value_cutoff
    : JoinedOrSeparate<["--"], "value-cutoff", [Show, Overlap]>,
      HelpText<"Cutoff value used for filtering. Meaning depends on subcommand">,
      MetaVarName<"<n>">;

def counts : Flag<["--"], "counts", [Show]>,
             HelpText<"Show counter values for shown functions">;
def show_format
    : JoinedOrSeparate<["--"], "show-format", [Show]>,
      HelpText<"Emit output in the selected format: text, json, or yaml">,
      MetaVarName<"<format>">;
def json : Flag<["--"], "json", [Show]>,
           HelpText<"Show sample profile data in JSON format "
                    "(deprecated, use --show-format=json)">;
def ic_targets : Flag<["--"], "ic-targets", [Show]>,
                 HelpText<"Show indirect call site target values">;
def show_vtables : Flag<["--"], "show-vtables", [Show]>,
                   HelpText<"Show vtable names for shown functions">;
def extbinary_write_vtable_type_prof
    : Flag<["--"], "extbinary-write-vtable-type-prof", [Merge]>,
      HelpText<"Include vtable type profile data in ext-binary sample profile">;

def memop_sizes : Flag<["--"], "memop-sizes", [Show]>,
                  HelpText<"Show profiled sizes of memory intrinsic calls">;
def detailed_summary : Flag<["--"], "detailed-summary", [Show]>,
                       HelpText<"Show detailed profile summary">;
def detailed_summary_cutoffs
    : JoinedOrSeparate<["--"], "detailed-summary-cutoffs", [Show]>,
      HelpText<"Cutoff percentages (times 10000) for generating detailed "
               "summary">,
      MetaVarName<"<list>">;
def hot_func_list : Flag<["--"], "hot-func-list", [Show]>,
                    HelpText<"Show profile summary of a list of hot functions">;
def all_functions : Flag<["--"], "all-functions", [Show]>,
                    HelpText<"Details for each and every function">;
def showcs : Flag<["--"], "showcs", [Show]>,
             HelpText<"Show context sensitive counts">;
def topn : JoinedOrSeparate<["--"], "topn", [Show]>,
           HelpText<"Show the list of functions with the largest internal counts">,
           MetaVarName<"<n>">;
def list_below_cutoff
    : Flag<["--"], "list-below-cutoff", [Show]>,
      HelpText<"Only output names of functions whose max count values are "
               "below the cutoff value">;
def show_prof_sym_list
    : Flag<["--"], "show-prof-sym-list", [Show]>,
      HelpText<"Show profile symbol list if it exists in the profile.">;
def show_sec_info_only
    : Flag<["--"], "show-sec-info-only", [Show]>,
      HelpText<"Show the information of each section in the sample profile "
               "(extbinary sample profiles only)">;
def binary_ids : Flag<["--"], "binary-ids", [Show]>,
                 HelpText<"Show binary ids in the profile.">;
def temporal_profile_traces
    : Flag<["--"], "temporal-profile-traces", [Show]>,
      HelpText<"Show temporal profile traces in the profile.">;
def covered : Flag<["--"], "covered", [Show]>,
              HelpText<"Show only the functions that have been executed.">;
def profile_version : Flag<["--"], "profile-version", [Show]>,
                      HelpText<"Show profile version.">;

def num_test_traces
    : JoinedOrSeparate<["--"], "num-test-traces", [Order]>,
      HelpText<"Keep aside the last <num-test-traces> traces when computing "
               "function order to evaluate that order">,
      MetaVarName<"<n>">;

def fs_discriminator_pass
    : JoinedOrSeparate<["--"], "fs-discriminator-pass",
                       [Show, Overlap, Merge]>,
      HelpText<"Zero out the discriminator bits for the FS discriminator "
               "pass beyond this value.">,
      MetaVarName<"<pass>">;
