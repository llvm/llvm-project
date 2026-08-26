include "llvm/Option/OptParser.td"

def LlcLocalOption : OptionFlag;

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--", "-"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--", "-"], name #"=">, HelpText<help>;
  def NAME : Separate<["--", "-"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

let Flags = [LlcLocalOption] in {

def help : FF<"help", "Display available options (-help-hidden for more)">;
def version : FF<"version", "Display the version">;

def O_flag : Joined<["-"], "O">, HelpText<"Optimization level. [-O0, -O1, -O2, or -O3] (default = '-O2')">, MetaVarName<"<n>">;

defm o : Eq<"o", "Output filename">, MetaVarName<"<filename>">;
defm x : Eq<"x", "Input language ('ir' or 'mir')">, MetaVarName<"<language>">;
defm mtriple : Eq<"mtriple", "Override target triple for module">, MetaVarName<"<target triple>">;
def M : JoinedOrSeparate<["-", "--"], "M">, HelpText<"InstPrinter options">, MetaVarName<"<options>">;

defm split_dwarf_output : Eq<"split-dwarf-output", ".dwo output filename">, MetaVarName<"<filename>">;
defm split_dwarf_file : Eq<"split-dwarf-file", "Specify the name of the .dwo file to encode in the DWARF output">, MetaVarName<"<filename>">;

defm time_compilations : Eq<"time-compilations", "Repeat compilation N times for timing">, MetaVarName<"<N>">;
def time_trace : FF<"time-trace", "Record time trace">;
defm time_trace_granularity : Eq<"time-trace-granularity", "Minimum time granularity (in microseconds) traced by time profiler">, MetaVarName<"<microseconds>">;
defm time_trace_file : Eq<"time-trace-file", "Specify time trace file destination">, MetaVarName<"<filename>">;

defm binutils_version : Eq<"binutils-version", "Produced object files can use all ELF features supported by this binutils version and newer.">, MetaVarName<"<version>">;
def preserve_as_comments : FF<"preserve-as-comments", "Preserve Comments in outputted assembly">;

def disable_verify : FF<"disable-verify", "Do not verify input module">;
def verify_each : FF<"verify-each", "Verify after each transform">;
def disable_simplify_libcalls : FF<"disable-simplify-libcalls", "Disable simplify-libcalls">;
def show_mc_encoding : FF<"show-mc-encoding", "Show encoding in .s output">;
defm output_asm_variant : Eq<"output-asm-variant", "Syntax variant to use for output printing">, MetaVarName<"<variant>">;
def dwarf_directory : FF<"dwarf-directory", "Use .file directives with an explicit directory">;
def dwarf_directory_EQ : JoinedOrSeparate<["-", "--"], "dwarf-directory=">, HelpText<"Use .file directives with an explicit directory">;
def asm_verbose : FF<"asm-verbose", "Add comments to directives.">;
def asm_verbose_EQ : Joined<["--", "-"], "asm-verbose=">, HelpText<"Add comments to directives.">;
def compile_twice : FF<"compile-twice", "Run everything twice, re-using the same pass manager and verify the result is the same.">;
def discard_value_names : FF<"discard-value-names", "Discard names from Value (other than GlobalValue).">;
def print_mir2vec_vocab : FF<"print-mir2vec-vocab", "Print MIR2Vec vocabulary contents">;
def print_mir2vec : FF<"print-mir2vec", "Print MIR2Vec embeddings for functions">;

def I : JoinedOrSeparate<["-"], "I">, HelpText<"include search path">, MetaVarName<"<dir>">;
def pass_remarks_with_hotness : FF<"pass-remarks-with-hotness", "With PGO, include profile count in optimization remarks">;
def pass_remarks_with_hotness_EQ : Joined<["--", "-"], "pass-remarks-with-hotness=">, HelpText<"With PGO, include profile count in optimization remarks">;
defm pass_remarks_hotness_threshold : Eq<"pass-remarks-hotness-threshold", "Minimum profile count required for an optimization remark to be output.">, MetaVarName<"<N or 'auto'>">;
defm pass_remarks_output : Eq<"pass-remarks-output", "Output filename for pass remarks">, MetaVarName<"<filename>">;
defm pass_remarks_filter : Eq<"pass-remarks-filter", "Only record optimization remarks from passes whose names match the given regular expression">, MetaVarName<"<regex>">;
defm pass_remarks_format : Eq<"pass-remarks-format", "The format used for serializing remarks (default: YAML)">, MetaVarName<"<format>">;

def load_pass_plugin_EQ : Joined<["--", "-"], "load-pass-plugin=">, HelpText<"Load plugin library">, MetaVarName<"<plugin>">;
def load_pass_plugin : Separate<["--", "-"], "load-pass-plugin">, Alias<load_pass_plugin_EQ>;

def enable_new_pm : FF<"enable-new-pm", "Enable the new pass manager">;
def enable_new_pm_EQ : Joined<["--", "-"], "enable-new-pm=">, HelpText<"Enable the new pass manager">;
defm passes : Eq<"passes", "A textual description of the pass pipeline.">, MetaVarName<"<pipeline>">;
def p : Separate<["--", "-"], "p">, Alias<passes_EQ>;
def p_EQ : Joined<["--", "-"], "p=">, Alias<passes_EQ>;

defm run_pass : Eq<"run-pass", "Run compiler only for specified passes (comma separated list)">, MetaVarName<"<pass-name>">;
defm pgo_kind : Eq<"pgo-kind", "The kind of profile guided optimization">, MetaVarName<"<kind>">;

} // end let Flags = [LlcLocalOption]


//
// NOTE: Each Opt declared below is forwarded to the backend. These are not explicitly used in llc.
//
def tail_merge_threshold : JoinedOrSeparate<["-", "--"], "tail-merge-threshold">, HelpText<"Threshold for tail merging">;
def pre_RA_sched : JoinedOrSeparate<["-", "--"], "pre-RA-sched">, HelpText<"Pre-RA Scheduler">;
def march : JoinedOrSeparate<["-", "--"], "march">, HelpText<"Architecture to generate code for">;
def mcpu : JoinedOrSeparate<["-", "--"], "mcpu">, HelpText<"Target a specific cpu type">;
def mattr : JoinedOrSeparate<["-", "--"], "mattr">, HelpText<"Target specific attributes">;
def relocation_model : JoinedOrSeparate<["-", "--"], "relocation-model">, HelpText<"Choose relocation model">;
def thread_model : JoinedOrSeparate<["-", "--"], "thread-model">, HelpText<"Choose threading model">;
def code_model : JoinedOrSeparate<["-", "--"], "code-model">, HelpText<"Choose code model">;
def large_data_threshold : JoinedOrSeparate<["-", "--"], "large-data-threshold">, HelpText<"Choose large data threshold">;
def exception_model : JoinedOrSeparate<["-", "--"], "exception-model">, HelpText<"Choose exception model">;
def frame_pointer : JoinedOrSeparate<["-", "--"], "frame-pointer">, HelpText<"Specify frame pointer elimination optimization">;
def denormal_fp_math : JoinedOrSeparate<["-", "--"], "denormal-fp-math">, HelpText<"Select denormal numbers mode">;
def denormal_fp_math_f32 : JoinedOrSeparate<["-", "--"], "denormal-fp-math-f32">, HelpText<"Select denormal numbers mode for float">;
def float_abi : JoinedOrSeparate<["-", "--"], "float-abi">, HelpText<"Choose float ABI type">;
def fp_contract : JoinedOrSeparate<["-", "--"], "fp-contract">, HelpText<"Enable aggressive formation of fused FP ops">;
def swift_async_fp : JoinedOrSeparate<["-", "--"], "swift-async-fp">, HelpText<"Swift async frame pointer mode">;
def trap_func : JoinedOrSeparate<["-", "--"], "trap-func">, HelpText<"Emit a call to trap function">;
def basic_block_sections : JoinedOrSeparate<["-", "--"], "basic-block-sections">, HelpText<"Emit basic blocks into separate sections">;
def tls_size : JoinedOrSeparate<["-", "--"], "tls-size">, HelpText<"Bit size of immediate TLS offsets">;
def meabi : JoinedOrSeparate<["-", "--"], "meabi">, HelpText<"Set EABI type">;
def debugger_tune : JoinedOrSeparate<["-", "--"], "debugger-tune">, HelpText<"Tune debug info for a particular debugger">;
def vector_library : JoinedOrSeparate<["-", "--"], "vector-library">, HelpText<"Vector functions library">;
def align_loops : JoinedOrSeparate<["-", "--"], "align-loops">, HelpText<"Default alignment for loops">;
def save_stats : FF<"save-stats", "Save LLVM statistics to a file">;
def save_stats_EQ : Joined<["-", "--"], "save-stats=">, HelpText<"Save LLVM statistics to a file">;
def filetype : JoinedOrSeparate<["-", "--"], "filetype">, HelpText<"Choose a file type">;
defm pass_remarks_missed : Eq<"pass-remarks-missed", "Enable missed optimization remarks from passes matching regex">, MetaVarName<"<regex>">;
defm pass_remarks_analysis : Eq<"pass-remarks-analysis", "Enable optimization remarks analysis from passes matching regex">, MetaVarName<"<regex>">;
defm pass_remarks : Eq<"pass-remarks", "Enable optimization remarks from passes matching regex">, MetaVarName<"<regex>">;
defm target_abi : Eq<"target-abi", "The name of the target ABI to use">, MetaVarName<"<name>">;

defm dwarf_version : Eq<"dwarf-version", "Dwarf version to use">, MetaVarName<"<version>">;
defm filter_print_funcs : Eq<"filter-print-funcs", "Only print IR for functions matching this">, MetaVarName<"<name>">;
defm print_after : Eq<"print-after", "Only print IR after these passes">, MetaVarName<"<pass-name>">;
defm print_before : Eq<"print-before", "Only print IR before these passes">, MetaVarName<"<pass-name>">;

// Forwarding-only options for separated fallback support.
def stop_before : JoinedOrSeparate<["-", "--"], "stop-before">, HelpText<"Stop before pass">;
def stop_after : JoinedOrSeparate<["-", "--"], "stop-after">, HelpText<"Stop after pass">;
def start_before : JoinedOrSeparate<["-", "--"], "start-before">, HelpText<"Start before pass">;
def start_after : JoinedOrSeparate<["-", "--"], "start-after">, HelpText<"Start after pass">;
def x86_asm_syntax : JoinedOrSeparate<["-", "--"], "x86-asm-syntax">, HelpText<"X86 assembly syntax mode">;
def info_output_file : JoinedOrSeparate<["-", "--"], "info-output-file">, HelpText<"File to append -stats and -timer output to">;
def gotol_abs_low_bound : JoinedOrSeparate<["-", "--"], "gotol-abs-low-bound">, HelpText<"BPF: Least number of PC-relative jump distance for GOTOL">;

// Backend-specific and global options for forwarding.
defm regalloc : Eq<"regalloc", "Register allocator to use">, MetaVarName<"<name>">;
defm debug_only : Eq<"debug-only", "Enable a specific type of debug output">, MetaVarName<"<type>">;
defm debug_pass : Eq<"debug-pass", "Print the name of each pass when it is executed">, MetaVarName<"<type>">;
def systemz_subreg_liveness : FF<"systemz-subreg-liveness", "Enable SystemZ subregister liveness tracking">;
def stress_sched : FF<"stress-sched", "Stress list scheduler">;
def terminal_rule : FF<"terminal-rule", "Apply the terminal rule during coalescing">;
def fast_isel : FF<"fast-isel", "Enable fast instruction selection">;
defm fast_isel_abort : Eq<"fast-isel-abort", "Enable/disable target-specific 'abort on fast-isel failure' mode">, MetaVarName<"<n>">;
defm global_isel_abort : Eq<"global-isel-abort", "Enable/disable target-specific 'abort on global-isel failure' mode">, MetaVarName<"<n>">;
def fast_isel_report_on_fallback : FF<"fast-isel-report-on-fallback", "Report when fast-isel falls back to SelectionDAG">;

// Boolean and value taking options that take no values causing fallthrough.
def verify_machineinstrs : FF<"verify-machineinstrs", "Verify machine instructions after each pass">;
def global_isel : FF<"global-isel", "Enable GlobalISel">;
def debugify_and_strip_all_safe : FF<"debugify-and-strip-all-safe", "Enable debugify and strip all safe">;
def debugify_and_strip_all_safe_EQ : Joined<["--", "-"], "debugify-and-strip-all-safe=">, HelpText<"Enable debugify and strip all safe">;
def simplify_mir : FF<"simplify-mir", "Simplify MIR output">;
def enable_arm_maskedldst : FF<"enable-arm-maskedldst", "Enable ARM masked load/store">;
def enable_arm_maskedgatscat : FF<"enable-arm-maskedgatscat", "Enable ARM masked gather/scatter">;
def force_dwarf_frame_section : FF<"force-dwarf-frame-section", "Force DWARF frame section">;
defm tail_predication : Eq<"tail-predication", "Tail predication mode">;
defm aarch64_cb_offset_bits : Eq<"aarch64-cb-offset-bits", "AArch64 CB offset bits">;
defm aarch64_enable_atomic_cfg_tidy : Eq<"aarch64-enable-atomic-cfg-tidy", "AArch64 enable atomic CFG tidy">;
defm aarch64_min_jump_table_entries : Eq<"aarch64-min-jump-table-entries", "AArch64 min jump table entries">;

