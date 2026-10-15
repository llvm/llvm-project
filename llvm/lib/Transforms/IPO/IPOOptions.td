//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def IPOOptions : OptionsStruct;

// BlockExtractor.cpp
defm : BoolField<"extract-blocks-erase-funcs", "false",
  "Erase the existing functions">;
defm : ValueField<"extract-blocks-file", "StringRef", "",
  "A file containing list of basic blocks to extract">;

// CalledValuePropagation.cpp
// The maximum number of functions to track per lattice value. Once the number
// of functions a call site can possibly target exceeds this threshold, it's
// lattice value becomes overdefined. The number of possible lattice values is
// bounded by Ch(F, M), where F is the number of functions in the module and M
// is this option's value. As such, this value should be kept very small. We
// likely can't do anything useful for call sites with a large number of
// possible targets, anyway.
defm : ValueField<"cvp-max-functions-per-value", "unsigned", "4",
  "The maximum number of functions to track per lattice value">;

// ElimAvailExtern.cpp
// This option was originally introduced to correctly support the lowering of
// LDS variables for AMDGPU when ThinLTO is enabled. It can be utilized for
// other purposes, but make sure it is safe to do so, as privatizing global
// variables is generally not safe.
defm : OptionalValueField<"avail-extern-gv-in-addrspace-to-local", "unsigned",
  "Convert available_externally global variables into locals if they are in specificed addrspace, renaming them to avoid link-time clashes">;
defm : BoolField<"avail-extern-to-local", "false",
  "Convert available_externally into locals, renaming them to avoid link-time clashes">;

// ExpandVariadics.cpp
defm : EnumField<"expand-variadics-override", "ExpandVariadicsMode", "",
  "unspecified,disable,optimize,lowering",
  ["ExpandVariadicsMode::Unspecified", "ExpandVariadicsMode::Disable", "ExpandVariadicsMode::Optimize", "ExpandVariadicsMode::Lowering"],
  "Override the behaviour of expand-variadics">;

// ForceFunctionAttrs.cpp
defm : ValueField<"forceattrs-csv-path", "StringRef", "",
  "Path to CSV file containing lines of function names and attributes to add to them in the form of `f1,attr1` or `f2,attr2=str`">;

// FunctionAttrs.cpp
defm : BoolField<"enable-poison-arg-attr-prop", "true",
  "Try to propagate nonnull and nofpclass argument attributes from callsites to caller functions">;
defm : BoolField<"nofree-inference", "true",
  "Infer nofree attribute during function-attrs pass">;
defm : BoolField<"nounwind-inference", "true",
  "Infer nounwind attribute during function-attrs pass">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-nofree-inference", "nofree-inference=false">;
def : AliasOf<"disable-nounwind-inference", "nounwind-inference=false">;

// FunctionImport.cpp
defm : BoolField<"enable-import-metadata", "false",
  "Enable import metadata like 'thinlto_src_module' and 'thinlto_src_file'">;
// Used when testing importing from distributed indexes via opt
// -function-import.
defm : BoolField<"import-all-index", "false",
  "Import all external functions in index">;
defm : BoolField<"print-imports", "false",
  "Print imported functions">;
// Summary file to use for function importing when using -function-import from
// the command line.
defm : ValueField<"summary-file", "StringRef", "",
  "The summary file to use for function importing">;

// FunctionSpecialization.cpp
defm : BoolField<"funcspec-for-literal-constant", "true",
  "Enable specialization of functions that take a literal constant as an argument">;
defm : BoolField<"funcspec-force", "false",
  "Force function specialization for every call site with a constant argument">;
defm : ValueField<"funcspec-max-block-predecessors", "unsigned", "2",
  "The maximum number of predecessors a basic block can have to be considered during the estimation of dead code">;
defm : ValueField<"funcspec-max-clones", "unsigned", "3",
  "The maximum number of clones allowed for a single function specialization">;
defm : ValueField<"funcspec-max-codesize-growth", "unsigned", "3",
  "Maximum codesize growth allowed per function">;
defm : ValueField<"funcspec-max-discovery-iterations", "unsigned", "100",
  "The maximum number of iterations allowed when searching for transitive phis">;
defm : ValueField<"funcspec-max-incoming-phi-values", "unsigned", "8",
  "The maximum number of incoming values a PHI node can have to be considered during the specialization bonus estimation">;
defm : ValueField<"funcspec-min-codesize-savings", "unsigned", "20",
  "Reject specializations whose codesize savings are less than this much percent of the original function size">;
defm : ValueField<"funcspec-min-function-size", "unsigned", "500",
  "Don't specialize functions that have less than this number of instructions">;
defm : ValueField<"funcspec-min-inlining-bonus", "unsigned", "300",
  "Reject specializations whose inlining bonus is less than this much percent of the original function size">;
defm : ValueField<"funcspec-min-latency-savings", "unsigned", "20",
  "Reject specializations whose latency savings are less than this much percent of the original function size">;
defm : BoolField<"funcspec-on-address", "false",
  "Enable function specialization on the address of global values">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"force-specialization", "funcspec-force">;

// GlobalDCE.cpp
defm : BoolField<"enable-vfe", "true",
  "Enable virtual function elimination">;

// GlobalOpt.cpp
defm : ValueField<"coldcc-rel-freq", "int", "2",
  "Maximum block frequency, expressed as a percentage of caller's entry frequency, for a call site to be considered cold for enabling coldcc">;
defm : BoolField<"enable-coldcc-stress-test", "false",
  "Enable stress test of coldcc by adding calling conv to all internal functions">;
defm : ValueField<"max-ifunc-versions", "unsigned", "5",
  "Maximum number of caller/callee versions that is allowed for using the expensive (cubic) static resolution algorithm">;
defm : BoolField<"optimize-non-fmv-callers", "true",
  "Statically resolve calls to versioned functions from non-versioned callers">;

// Inliner.cpp
defm : ValueField<"cgscc-inline-replay", "StringRef", "",
  "Optimization remarks file containing inline remarks to be replayed by cgscc inlining">;
defm : EnumField<"cgscc-inline-replay-fallback", "ReplayInlinerSettings::Fallback", "",
  "Original,AlwaysInline,NeverInline",
  ["ReplayInlinerSettings::Fallback::Original", "ReplayInlinerSettings::Fallback::AlwaysInline", "ReplayInlinerSettings::Fallback::NeverInline"],
  "How cgscc inline replay treats sites that don't come from the replay. Original: defers to original advisor, AlwaysInline: inline all sites not in replay, NeverInline: inline no sites not in replay">;
defm : EnumField<"cgscc-inline-replay-format", "CallSiteFormat::Format", "CallSiteFormat::Format::LineColumnDiscriminator",
  "Line,LineColumn,LineDiscriminator,LineColumnDiscriminator",
  ["CallSiteFormat::Format::Line", "CallSiteFormat::Format::LineColumn", "CallSiteFormat::Format::LineDiscriminator", "CallSiteFormat::Format::LineColumnDiscriminator"],
  "How cgscc inline replay file is formatted">;
defm : EnumField<"cgscc-inline-replay-scope", "ReplayInlinerSettings::Scope", "",
  "Function,Module",
  ["ReplayInlinerSettings::Scope::Function", "ReplayInlinerSettings::Scope::Module"],
  "Whether inline replay should be applied to the entire Module or just the Functions (default) that are present as callers in remarks during cgscc inlining">;
defm : ValueField<"inliner-forwarding-scan-limit", "unsigned", "16",
  "Maximum number of instructions to scan backward for store-to-load forwarding in subsequent inlining decisions. DefMaxInstsToScan=6 is not enough and misses inlining opportunities (e.g. when class stores into mutiple members in ctor and afterwards calls a function reading those members)">;
defm : ValueField<"intra-scc-cost-multiplier", "int", "2",
  "Cost multiplier to multiply onto inlined call sites where the new call was previously an intra-SCC call (not relevant when the original call was already intra-SCC). This can accumulate over multiple inlinings (e.g. if a call site already had a cost multiplier and one of its inlined calls was also subject to this, the inlined call would have the original multiplier multiplied by intra-scc-cost-multiplier). This is to prevent tons of inlining through a child SCC which can cause terrible compile times">;
// A flag for test, so we can print the content of the advisor when running it
// as part of the default (e.g. -O3) pipeline.
defm : BoolField<"keep-inline-advisor-for-printing", "false", "">;

// Instrumentor.cpp
defm : ValueField<"instrumentor-read-config-paths-file", "StringRef", "",
  "Read the instrumentor configuration file paths from the specified file (newline separated)">;
defm : ValueField<"instrumentor-write-config-file", "StringRef", "",
  "Write the instrumentor configuration into the specified JSON file">;

// LowerTypeTests.cpp
defm : BoolField<"lowertypetests-avoid-reuse", "true",
  "Try to avoid reuse of byte array addresses using aliases">;
// FIXME: Remove in clang 24.
defm : BoolField<"lowertypetests-jump-table-debug-info", "true",
  "Enable debug info generation for jump tables">;
defm : ValueField<"lowertypetests-read-summary", "StringRef", "",
  "Read summary from given textual assembly or YAML file before running pass">;
defm : EnumField<"lowertypetests-summary-action", "PassSummaryAction", "",
  "none,import,export",
  ["PassSummaryAction::None", "PassSummaryAction::Import", "PassSummaryAction::Export"],
  "What to do with the summary when running this pass">;
defm : ValueField<"lowertypetests-write-summary", "StringRef", "",
  "Write summary to given YAML file after running pass">;
// FIXME: Remove in clang 26.
defm : BoolField<"reorder-cfi-jump-tables-profiles", "true",
  "Reorder CFI jump tables using profile information">;

// MergeFunctions.cpp
// Under option -mergefunc-preserve-debug-info we:
// - Do not create a new function for a thunk.
// - Retain the debug info for a thunk's parameters (and associated
//   instructions for the debug info) from the entry block.
//   Note: -debug will display the algorithm at work.
// - Create debug-info for the call (to the shared implementation) made by
//   a thunk and its return value.
// - Erase the rest of the function, retaining the (minimally sized) entry
//   block to create a thunk.
// - Preserve a thunk's call site to point to the thunk even when both occur
//   within the same translation unit, to aid debugability. Note that this
//   behaviour differs from the underlying -mergefunc implementation which
//   modifies the thunk's call site to point to the shared implementation
//   when both occur within the same translation unit.
defm : BoolField<"mergefunc-preserve-debug-info", "false",
  "Preserve debug info in thunk when mergefunc transformations are made">;
defm : BoolField<"mergefunc-use-aliases", "false",
  "Allow mergefunc to create aliases">;
let Condition = AssertsOnly in
defm : ValueField<"mergefunc-verify", "unsigned", "0",
  "How many functions in a module could be used for MergeFunctions to pass a basic correctness check. '0' disables this check. Works only with '-debug' key">;

// ModuleInliner.cpp
defm : BoolField<"ctx-prof-promote-alwaysinline", "false",
  "If using a contextual profile in this module, and an indirect call target is marked as alwaysinline, perform indirect call promotion for that target. If multiple targets for an indirect call site fit this description, they are all promoted">;

// PartialInlining.cpp
// Used to determine when an edge is considered cold. Default is set to 10%. ie.
// if the branch probability is 10% or less, then it is deemed as 'cold'.
defm : ValueField<"cold-branch-ratio", "float", "0.1f",
  "Minimum BranchProbability to consider a region cold">;
defm : ValueField<"max-num-inline-blocks", "unsigned", "5",
  "Max number of blocks to be partially inlined">;
defm : ValueField<"max-partial-inlining", "int", "-1",
  "Max number of partial inlining. The default is unlimited">;
// Used to tune the minimum number of execution counts needed in the predecessor
// block to the cold edge. ie. confidence interval.
defm : ValueField<"min-block-execution", "unsigned", "100",
  "Minimum block executions to consider its BranchProbabilityInfo valid">;
// Used to determine if a cold region is worth outlining based on
// its inlining cost compared to the original function.  Default is set at 10%.
// ie. if the cold region reduces the inlining cost of the original function by
// at least 10%.
defm : ValueField<"min-region-size-ratio", "float", "0.1f",
  "Minimum ratio comparing relative sizes of each outline candidate and original function">;
defm : BoolField<"mr-partial-inlining", "true",
  "Enable multi-region partial inlining">;
// Used only when PGO or user annotated branch data is absent. It is
// the least value that is used to weigh the outline region. If BFI
// produces larger value, the BFI value will be used.
defm : ValueField<"outline-region-freq-percent", "int", "75",
  "Relative frequency of outline region to the entry block">;
defm : BoolField<"partial-inlining", "true",
  "Enable partial inlining">;
defm : ValueField<"partial-inlining-extra-penalty", "unsigned", "0",
  "A debug option to add additional penalty to the computed one">;
defm : BoolField<"pi-force-live-exit-outline", "false",
  "Force outline regions with live exits">;
defm : BoolField<"pi-mark-coldcc", "false",
  "Mark outline function calls with ColdCC">;
// This is an option used by testing:
defm : BoolField<"skip-partial-inlining-cost-analysis", "false",
  "Skip Cost Analysis">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-mr-partial-inlining", "mr-partial-inlining=false">;
def : AliasOf<"disable-partial-inlining", "partial-inlining=false">;

// SCCP.cpp
defm : ValueField<"funcspec-max-iters", "unsigned", "10",
  "The maximum number of iterations function specialization is run">;

// SampleProfile.cpp
defm : BoolField<"annotate-sample-profile-inline-phase", "false",
  "Annotate LTO phase (prelink / postlink), or main (no LTO) for sample-profile inline pass name">;
defm : ValueField<"hot-func-cutoff-for-staleness-error", "unsigned", "800000",
  "A function is considered hot for staleness error check if its total sample count is above the specified percentile">;
defm : ValueField<"min-functions-for-staleness-error", "unsigned", "50",
  "Skip the check if the number of hot functions is smaller than the specified number">;
defm : BoolField<"overwrite-existing-weights", "false",
  "Ignore existing branch weights on IR and always overwrite">;
defm : ValueField<"percent-mismatch-for-staleness-error", "unsigned", "80",
  "Reject the profile if the mismatch percent is higher than the given number">;
defm : BoolField<"persist-profile-staleness", "false",
  "Compute stale profile statistical metrics and write it into the native object file(.llvm_stats section)">;
defm : BoolField<"profile-accurate-for-symsinlist", "true",
  "For symbols in profile symbol list, regard their profiles to be accurate. It may be overridden by profile-sample-accurate">;
defm : BoolField<"profile-sample-accurate", "false",
  "If the sample profile is accurate, we will mark all un-sampled callsite and function as having 0 samples. Otherwise, treat un-sampled callsites and functions conservatively as unknown">;
defm : BoolField<"profile-sample-block-accurate", "false",
  "If the sample profile is accurate, we will mark all un-sampled branches and calls as having 0 samples. Otherwise, treat them conservatively as unknown">;
defm : BoolField<"report-profile-staleness", "false",
  "Compute and report stale profile statistical metrics">;
defm : OptionalBoolField<"salvage-stale-profile",
  "Salvage stale profile by fuzzy matching and use the remapped location for sample profile query">;
defm : OptionalBoolField<"salvage-unused-profile",
  "Salvage unused profile by matching with new functions on call graph">;
// Since profiles are consumed by many passes, turning off this option has
// side effects. For instance, pre-link SCC inliner would see merged profiles
// and inline the hot functions (that are skipped in this pass).
defm : OptionalBoolField<"sample-loader-inlining",
  "If false, artificially skip inline transformation in sample-loader pass, and merge (or scale) profiles (as configured by --sample-profile-merge-inlinee)">;
// Command line option to specify the file to read samples from. This is
// mainly used for debugging.
defm : ValueField<"sample-profile-file", "StringRef", "",
  "Profile file loaded by -sample-profile">;
defm : ValueField<"sample-profile-icp-max-prom", "unsigned", "3",
  "Max number of promotions for a single indirect call callsite in sample profile loader">;
defm : ValueField<"sample-profile-icp-relative-hotness", "unsigned", "25",
  "Relative hotness percentage threshold for indirect call promotion in priority-based sample profile loader inlining">;
defm : ValueField<"sample-profile-icp-relative-hotness-skip", "unsigned", "1",
  "Skip relative hotness check for ICP up to given number of targets">;
defm : ValueField<"sample-profile-inline-replay", "StringRef", "",
  "Optimization remarks file containing inline remarks to be replayed by inlining from sample profile loader">;
defm : EnumField<"sample-profile-inline-replay-fallback", "ReplayInlinerSettings::Fallback", "",
  "Original,AlwaysInline,NeverInline",
  ["ReplayInlinerSettings::Fallback::Original", "ReplayInlinerSettings::Fallback::AlwaysInline", "ReplayInlinerSettings::Fallback::NeverInline"],
  "How sample profile inline replay treats sites that don't come from the replay. Original: defers to original advisor, AlwaysInline: inline all sites not in replay, NeverInline: inline no sites not in replay">;
defm : EnumField<"sample-profile-inline-replay-format", "CallSiteFormat::Format", "CallSiteFormat::Format::LineColumnDiscriminator",
  "Line,LineColumn,LineDiscriminator,LineColumnDiscriminator",
  ["CallSiteFormat::Format::Line", "CallSiteFormat::Format::LineColumn", "CallSiteFormat::Format::LineDiscriminator", "CallSiteFormat::Format::LineColumnDiscriminator"],
  "How sample profile inline replay file is formatted">;
defm : EnumField<"sample-profile-inline-replay-scope", "ReplayInlinerSettings::Scope", "",
  "Function,Module",
  ["ReplayInlinerSettings::Scope::Function", "ReplayInlinerSettings::Scope::Module"],
  "Whether inline replay should be applied to the entire Module or just the Functions (default) that are present as callers in remarks during sample profile inlining">;
defm : OptionalBoolField<"sample-profile-inline-size",
  "Inline cold call sites in profile loader if it's beneficial for code size">;
defm : BoolField<"sample-profile-merge-inlinee", "true",
  "Merge past inlinee's profile to outline version if sample profile loader decided not to inline a call site. It will only be enabled when top-down order of profile loading is enabled">;
defm : OptionalBoolField<"sample-profile-prioritized-inline",
  "Use call site prioritized inlining for sample profile loader. Currently only CSSPGO is supported">;
defm : OptionalBoolField<"sample-profile-recursive-inline",
  "Allow sample loader inliner to inline recursive calls">;
// The named file contains a set of transformations that may have been applied
// to the symbol names between the program from which the sample data was
// collected and the current program's symbols.
defm : ValueField<"sample-profile-remapping-file", "StringRef", "",
  "Profile remapping file loaded by -sample-profile">;
defm : BoolField<"sample-profile-remove-probe", "false",
  "Remove pseudo-probe after sample profile annotation">;
defm : BoolField<"sample-profile-top-down-load", "true",
  "Do profile annotation and inlining for functions in top-down order of call graph during sample profile loading. It only works for new pass manager">;
defm : OptionalBoolField<"sample-profile-use-preinliner",
  "Use the preinliner decisions stored in profile context">;
defm : BoolField<"use-profiled-call-graph", "true",
  "Process functions in a top-down order defined by the profiled call graph when -sample-profile-top-down-load is on">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-sample-loader-inlining", "sample-loader-inlining=false">;
def : AliasOf<"precent-mismatch-for-staleness-error", "percent-mismatch-for-staleness-error">;

// SampleProfileMatcher.cpp
defm : ValueField<"func-profile-similarity-threshold", "unsigned", "80",
  "Consider a profile matches a function if the similarity of their callee sequences is above the specified percentile">;
defm : BoolField<"load-func-profile-for-cg-matching", "true",
  "Load top-level profiles that the sample reader initially skipped for the call-graph matching (only meaningful for extended binary format)">;
defm : ValueField<"min-call-count-for-cg-matching", "unsigned", "3",
  "The minimum number of call anchors required for a function to run stale profile call graph matching">;
defm : ValueField<"min-func-count-for-cg-matching", "unsigned", "5",
  "The minimum number of basic blocks required for a function to run stale profile call graph matching">;
defm : ValueField<"salvage-stale-profile-max-callsites", "unsigned", "std::numeric_limits<unsigned>::max()",
  "The maximum number of callsites in a function, above which stale profile matching will be skipped">;
defm : ValueField<"salvage-unused-profile-max-functions", "unsigned", "std::numeric_limits<unsigned>::max()",
  "The maximum number of functions in a module, above which salvage unused profile will be skipped">;

// SampleProfileProbe.cpp
defm : BoolField<"update-pseudo-probe", "true",
  "Update pseudo probe distribution factor">;
defm : BoolField<"verify-pseudo-probe", "false",
  "Do pseudo probe verification">;
defm : ListField<"verify-pseudo-probe-funcs", "StringRef",
  "The option to specify the name of the functions to verify">;

// StripSymbols.cpp
defm : BoolField<"strip-global-constants", "false",
  "Removes debug compile units which reference to non-existing global constants">;

// WholeProgramDevirt.cpp
defm : BoolField<"devirtualize-speculatively", "false",
  "Enable speculative devirtualization optimization">;
defm : ValueField<"wholeprogramdevirt-branch-funnel-threshold", "unsigned", "10",
  "Maximum number of call targets per call site to enable branch funnels">;
defm : EnumField<"wholeprogramdevirt-check", "WPDCheckMode", "",
  "none,trap,fallback",
  ["WPDCheckMode::None", "WPDCheckMode::Trap", "WPDCheckMode::Fallback"],
  "Type of checking for incorrect devirtualizations">;
defm : ValueField<"wholeprogramdevirt-read-summary", "StringRef", "",
  "Read summary from given bitcode or YAML file before running pass">;
defm : EnumField<"wholeprogramdevirt-summary-action", "PassSummaryAction", "",
  "none,import,export",
  ["PassSummaryAction::None", "PassSummaryAction::Import", "PassSummaryAction::Export"],
  "What to do with the summary when running this pass">;
defm : ValueField<"wholeprogramdevirt-write-summary", "StringRef", "",
  "Write summary to given bitcode or YAML file after running pass. Output file format is deduced from extension: *.bc means writing bitcode, otherwise YAML">;
