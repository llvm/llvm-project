//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def AnalysisOptions : OptionsStruct;

// AliasAnalysis.cpp
// Print a trace of alias analysis queries and their results.
let Condition = AssertsOnly in
defm : BoolField<"aa-trace", "false", "">;
// Allow disabling BasicAA from the AA results. This is particularly useful when
// testing to isolate a single AA implementation.
defm : BoolField<"disable-basic-aa", "false", "">;

// AliasAnalysisEvaluator.cpp
defm : BoolField<"evaluate-aa-metadata", "false", "">;
defm : BoolField<"print-all-alias-modref-info", "false", "">;
defm : BoolField<"print-may-aliases", "false", "">;
defm : BoolField<"print-mod", "false", "">;
defm : BoolField<"print-modref", "false", "">;
defm : BoolField<"print-must-aliases", "false", "">;
defm : BoolField<"print-no-aliases", "false", "">;
defm : BoolField<"print-no-modref", "false", "">;
defm : BoolField<"print-partial-aliases", "false", "">;
defm : BoolField<"print-ref", "false", "">;

// AssumptionCache.cpp
defm : ValueField<"max-assumes-per-value", "unsigned", "1024",
  "Maximum number of assumptions to cache for a single value">;
defm : BoolField<"verify-assumption-cache", "false",
  "Enable verification of assumption cache">;

// BasicAliasAnalysis.cpp
// Enable analysis of recursive PHI nodes.
defm : BoolField<"basic-aa-recphi", "true", "">;
defm : BoolField<"basic-aa-separate-storage", "true", "">;

// BlockFrequencyInfo.cpp
defm : BoolField<"print-bfi", "false",
  "Print the block frequency info">;
defm : ValueField<"view-bfi-func-name", "StringRef", "",
  "The option to specify the name of the function whose CFG will be displayed">;
defm : ValueField<"view-hot-freq-percent", "unsigned", "10",
  "An integer in percent used to specify the hot blocks/edges to be displayed in red: a block or edge whose frequency is no less than the max frequency of the function multiplied by this percent">;
defm : EnumField<"pgo-view-counts", "PGOViewCountsType", "PGOVCT_None",
  "none,graph,text",
  ["PGOVCT_None", "PGOVCT_Graph", "PGOVCT_Text"],
  "A boolean option to show CFG dag or text with block profile counts and branch probabilities right after PGO profile annotation step. The profile counts are computed using branch probabilities from the runtime profile data and block frequency propagation algorithm. To view the raw counts from the profile, use option -pgo-view-raw-counts instead. To limit graph display to only one function, use filtering option -view-bfi-func-name">;
defm : ValueField<"print-bfi-func-name", "StringRef", "",
  "The option to specify the name of the function whose block frequency info is printed">;

// BlockFrequencyInfoImpl.cpp
defm : BoolField<"check-bfi-unknown-block-queries", "false",
  "Check if block frequency is queried for an unknown block for debugging missed BFI updates">;
defm : ValueField<"iterative-bfi-max-iterations-per-block", "unsigned", "1000",
  "Iterative inference: maximum number of update iterations per block">;
defm : ValueField<"iterative-bfi-precision", "double", "1e-12",
  "Iterative inference: delta convergence precision; smaller values typically lead to better results at the cost of worsen runtime">;

// BranchProbabilityInfo.cpp
defm : BoolField<"print-bpi", "false",
  "Print the branch probability info">;
defm : ValueField<"print-bpi-func-name", "StringRef", "",
  "The option to specify the name of the function whose branch probability info is printed">;

// CFGPrinter.cpp
defm : ValueField<"cfg-dot-filename-prefix", "StringRef", "",
  "The prefix used for the CFG dot file names">;
defm : ValueField<"cfg-func-name", "StringRef", "",
  "The name of a function (or its substring) whose CFG is viewed/printed">;
defm : BoolField<"cfg-heat-colors", "true",
  "Show heat colors in CFG">;
defm : OptionalValueField<"cfg-hide-cold-paths", "double",
  "Hide blocks with relative frequency below the given value">;
defm : BoolField<"cfg-hide-deoptimize-paths", "false", "">;
defm : BoolField<"cfg-hide-unreachable-paths", "false", "">;
defm : BoolField<"cfg-raw-weights", "false",
  "Use raw weights for labels. Use percentages as default">;
defm : BoolField<"cfg-weights", "false",
  "Show edges labeled with weights">;

// CGSCCPassManager.cpp
defm : BoolField<"abort-on-max-devirt-iterations-reached", "false",
  "Abort when the max iterations for devirtualization CGSCC repeat pass is reached">;

// CallGraphSCCPass.cpp
defm : ValueField<"max-devirt-iterations", "unsigned", "4", "">;

// CallPrinter.cpp
defm : ValueField<"callgraph-dot-filename-prefix", "StringRef", "",
  "The prefix used for the CallGraph dot file names">;
// This option shows static (relative) call counts. FIXME: Need to show real
// counts when profile data is available.
defm : BoolField<"callgraph-heat-colors", "false",
  "Show heat colors in call-graph">;
defm : BoolField<"callgraph-multigraph", "false",
  "Show call-multigraph (do not remove parallel edges)">;
defm : BoolField<"callgraph-show-weights", "false",
  "Show edges labeled with weights">;

// CostModel.cpp
defm : EnumField<"cost-kind", "OutputCostKind", "OutputCostKind::RecipThroughput", "throughput,latency,code-size,size-latency,all", ["OutputCostKind::RecipThroughput", "OutputCostKind::Latency", "OutputCostKind::CodeSize", "OutputCostKind::SizeAndLatency", "OutputCostKind::All"],
  "Target cost kind">;
defm : EnumField<"intrinsic-cost-strategy", "IntrinsicCostStrategy", "IntrinsicCostStrategy::InstructionCost", "instruction-cost,intrinsic-cost,type-based-intrinsic-cost", ["IntrinsicCostStrategy::InstructionCost", "IntrinsicCostStrategy::IntrinsicCost", "IntrinsicCostStrategy::TypeBasedIntrinsicCost"],
  "Costing strategy for intrinsic instructions">;

// CtxProfAnalysis.cpp
defm : OptionalBoolField<"ctx-profile-force-is-specialized",
  "Treat the given module as-if it were containing the post-thinlink module containing the root">;

// DDG.cpp
defm : BoolField<"ddg-pi-blocks", "true",
  "Create pi-block nodes">;
defm : BoolField<"ddg-simplify", "true",
  "Simplify DDG by merging nodes that have less interesting edges">;

// DDGPrinter.cpp
defm : ValueField<"dot-ddg-filename-prefix", "StringRef", "\"ddg\"",
  "The prefix used for the DDG dot file names">;
defm : BoolField<"dot-ddg-only", "false",
  "simple ddg dot graph">;

// Delinearization.cpp
defm : BoolField<"delinearize-use-fixed-size-array-heuristic", "true",
  "When printing analysis, use the heuristic for fixed-size arrays if the default delinearization fails">;

// DependenceAnalysis.cpp
defm : BoolField<"da-delinearization-checks", "true",
  "Check that delinearized subscripts are statically valid. Disabling this option may result in incorrect dependence vectors for languages that allow the subscript of one dimension to underflow or overflow into another dimension">;
defm : BoolField<"da-delinearize", "true",
  "Try to delinearize array references">;
defm : EnumField<"da-enable-dependence-test", "DependenceTestType", "DependenceTestType::Default", "default,all,strong-siv,weak-crossing-siv,exact-siv,weak-zero-siv,exact-rdiv,gcd-miv,banerjee-miv", ["DependenceTestType::Default", "DependenceTestType::All", "DependenceTestType::StrongSIV", "DependenceTestType::WeakCrossingSIV", "DependenceTestType::ExactSIV", "DependenceTestType::WeakZeroSIV", "DependenceTestType::ExactRDIV", "DependenceTestType::GCDMIV", "DependenceTestType::BanerjeeMIV"],
  "Run only specified dependence test routine and disable others. The purpose is mainly to exclude the influence of other dependence test routines in regression tests. If set to All, all dependence test routines are enabled">;
defm : ValueField<"da-miv-max-level-threshold", "unsigned", "7",
  "Maximum depth allowed for the recursive algorithm used to explore MIV direction vectors">;

// FunctionPropertiesAnalysis.cpp
defm : ValueField<"big-basic-block-instruction-threshold", "unsigned", "500",
  "The minimum number of instructions a basic block should contain before being considered big">;
defm : ValueField<"call-with-many-arguments-threshold", "unsigned", "4",
  "The minimum number of arguments a function call must have before it is considered having many arguments">;
defm : ValueField<"medium-basic-block-instruction-threshold", "unsigned", "15",
  "The minimum number of instructions a basic block should contain before being considered medium-sized">;

// GlobalsModRef.cpp
// An option to enable unsafe alias results from the GlobalsModRef analysis.
// When enabled, GlobalsModRef will provide no-alias results which in extremely
// rare cases may not be conservatively correct. In particular, in the face of
// transforms which cause asymmetry between how effective getUnderlyingObject is
// for two pointers, it may produce incorrect results. These unsafe results
// have been returned by GMR for many years without causing significant issues
// in the wild and so we provide a mechanism to re-enable them for users of LLVM
// that have a particular performance sensitivity and no known issues. The
// option also makes it easy to evaluate the performance impact of these
// results.
defm : BoolField<"enable-unsafe-globalsmodref-alias-results", "false", "">;

// IndirectCallPromotionAnalysis.cpp
// Set the maximum number of targets to promote for a single indirect-call
// callsite.
defm : ValueField<"icp-max-num-vtables", "unsigned", "6",
  "Max number of vtables annotated for a vtable load instruction">;
defm : ValueField<"icp-max-prom", "unsigned", "3",
  "Max number of promotions for a single indirect call callsite">;
// Set the minimum absolute count threshold for indirect call promotion.
// Candidates with counts below this threshold will not be promoted.
defm : ValueField<"icp-minimum-count-threshold", "unsigned", "0",
  "Minimum absolute count for promotion candidate">;
// The percent threshold for the direct-call target (this call site vs the
// remaining call count) for it to be considered as the promotion target.
defm : ValueField<"icp-remaining-percent-threshold", "unsigned", "30",
  "The percentage threshold against remaining unpromoted indirect call count for the promotion">;
// The percent threshold for the direct-call target (this call site vs the total
// call count) for it to be considered as the promotion target.
defm : ValueField<"icp-total-percent-threshold", "uint64_t", "5",
  "The percentage threshold against total count for the promotion">;

// InlineAdvisor.cpp
defm : BoolField<"annotate-inline-phase", "false",
  "If true, annotate inline advisor remarks with LTO and pass information">;
defm : BoolField<"inline-deferral", "false",
  "Enable deferred inlining">;
// An integer used to limit the cost of inline deferral. The default negative
// number tells shouldBeDeferred to only take the secondary cost into account.
defm : ValueField<"inline-deferral-scale", "int", "2",
  "Scale to limit the cost of inline deferral">;
// Flag to add inline messages as callsite attributes 'inline-remark'.
defm : BoolField<"inline-remark-attribute", "false",
  "Enable adding inline-remark attribute to callsites processed by inliner but decided to be not inlined">;
// This flag is used to enable IR2Vec embeddings in the ML inliner; Only valid
// with ML inliner. The vocab file is used to initialize the embeddings.
defm : ValueField<"ml-inliner-ir2vec-vocab-file", "StringRef", "",
  "Vocab file for IR2Vec; Setting this enables configuring the model to use IR2Vec embeddings">;

// InlineCost.cpp
defm : ValueField<"cold-callsite-rel-freq", "int", "2",
  "Maximum block frequency, expressed as a percentage of caller's entry frequency, for a callsite to be cold in the absence of profile information">;
defm : BoolField<"gep-const-evaluation", "true",
  "Evaluate GetElementPtr with constant operands">;
defm : ValueField<"hot-callsite-rel-freq", "uint64_t", "60",
  "Minimum block frequency, expressed as a multiple of caller's entry frequency, for a callsite to be hot in the absence of profile information">;
// We introduce this option since there is a minor compile-time win by avoiding
// addition of TTI attributes (target-features in particular) to inline
// candidates when they are guaranteed to be the same as top level methods in
// some use cases. If we avoid adding the attribute, we need an option to avoid
// checking these attributes.
defm : BoolField<"ignore-tti-inline-compatible", "false",
  "Ignore TTI attributes compatibility check between callee/caller during inline cost calculation">;
defm : BoolField<"inline-all-viable-calls", "false",
  "Inline all viable calls, even if they exceed the inlining threshold">;
defm : ValueField<"inline-asm-instr-cost", "int", "0",
  "Cost of a single inline asm instruction when inlining">;
defm : ValueField<"inline-call-penalty", "int", "25",
  "Call penalty that is applied per callsite when inlining">;
defm : BoolField<"inline-caller-superset-nobuiltin", "true",
  "Allow inlining when caller has a superset of callee's nobuiltin attributes">;
defm : BoolField<"inline-cost-full", "false",
  "Compute the full inline cost of a call site even when the cost exceeds the threshold">;
defm : OptionalBoolField<"inline-enable-cost-benefit-analysis",
  "Enable the cost-benefit analysis for the inliner">;
defm : OptionalValueField<"inline-max-stacksize", "size_t",
  "Do not inline functions with a stack size that exceeds the specified limit">;
defm : ValueField<"inline-memaccess-cost", "int", "0",
  "Cost of load/store instruction when inlining">;
// -inline-savings-multiplier overrides per TTI multipliers iff it is specified
// explicitly in command line options. This option is exposed for tuning and
// testing.
defm : OptionalValueField<"inline-savings-multiplier", "int",
  "Multiplier to multiply cycle savings by during inlining">;
// -inline-savings-profitable-multiplier overrides per TTI multipliers iff it is
// specified explicitly in command line options. This option is exposed for
// tuning and testing.
defm : OptionalValueField<"inline-savings-profitable-multiplier", "int",
  "A multiplier on top of cycle savings to decide whether the savings won't justify the cost">;
defm : ValueField<"inline-size-allowance", "int", "100",
  "The maximum size of a callee that get's inlined without sufficient cycle savings">;
defm : BoolField<"print-instruction-comments", "false",
  "Prints comments for instruction based on inline cost analysis">;
defm : ValueField<"recursive-inline-max-stacksize", "size_t", "1024",
  "Do not inline recursive functions with a stack size that exceeds the specified limit">;

// InlineOrder.cpp
defm : EnumField<"inline-priority-mode", "InlinePriorityMode", "InlinePriorityMode::Size", "size,cost,cost-benefit,ml", ["InlinePriorityMode::Size", "InlinePriorityMode::Cost", "InlinePriorityMode::CostBenefit", "InlinePriorityMode::ML"],
  "Choose the priority mode to use in module inline">;
defm : ValueField<"module-inliner-top-priority-threshold", "int", "0",
  "The cost threshold for call sites that get inlined without the cost-benefit analysis">;

// InstructionPrecedenceTracking.cpp
let Condition = AssertsOnly in
defm : BoolField<"ipt-expensive-asserts", "false",
  "Perform expensive assert validation on every query to Instruction Precedence Tracking">;

// InteractiveModelRunner.cpp
defm : BoolField<"interactive-model-runner-echo-reply", "false",
  "The InteractiveModelRunner will echo back to stderr the data received from the host (for debugging purposes)">;

// LazyValueInfo.cpp
defm : BoolField<"lvi-per-pred-ranges", "false",
  "Enable tracking of ranges for a value in a block for each block predecessor">;

// LoopAccessAnalysis.cpp
// This enables versioning on the strides of symbolically striding memory
// accesses in code like the following.
//   for (i = 0; i < N; ++i)
//     A[i * Stride1] += B[i * Stride2] ...
//
// Will be roughly translated to
//    if (Stride1 == 1 && Stride2 == 1) {
//      for (i = 0; i < N; i+=4)
//       A[i:i+3] += ...
//    } else
//      ...
defm : BoolField<"enable-mem-access-versioning", "true",
  "Enable symbolic stride memory access versioning">;
defm : BoolField<"laa-speculate-unit-stride", "true",
  "Speculate that non-constant strides are unit in LAA">;
// We collect dependences up to this threshold.
defm : ValueField<"max-dependences", "unsigned", "100",
  "Maximum number of dependences collected by loop-access analysis">;
defm : ValueField<"max-forked-scev-depth", "unsigned", "5",
  "Maximum recursion depth when finding forked SCEVs">;
// The maximum iterations used to merge memory checks.
defm : ValueField<"memory-check-merge-threshold", "unsigned", "100",
  "Maximum number of comparisons done when trying to merge runtime memory checks">;
defm : ValueField<"stencil-merge-max-groups", "unsigned", "4096",
  "Skip stencil group merging when the number of runtime checking groups exceeds this limit, to bound compile time">;
defm : EnumField<"stencil-runtime-check-merge", "StencilMergePolicy", "StencilMergePolicy::Off", "off,auto,force", ["StencilMergePolicy::Off", "StencilMergePolicy::Auto", "StencilMergePolicy::Force"],
  "Control stencil-pattern merging of runtime memory checks">;
// Enable store-to-load forwarding conflict detection. This option can be
// disabled for correctness testing.
defm : BoolField<"store-to-load-forwarding-conflict-detection", "true",
  "Enable conflict detection in loop-access analysis">;

// LoopCacheAnalysis.cpp
defm : ValueField<"default-trip-count", "unsigned", "100",
  "Use this to specify the default trip count of a loop">;
// In this analysis two array references are considered to exhibit temporal
// reuse if they access either the same memory location, or a memory location
// with distance smaller than a configurable threshold.
defm : ValueField<"temporal-reuse-threshold", "unsigned", "2",
  "Use this to specify the max. distance between array elements accessed in a loop so that the elements are classified to have temporal reuse">;

// MLInlineAdvisor.cpp
defm : ValueField<"inliner-interactive-channel-base", "StringRef", "",
  "Base file path for the interactive mode. The incoming filename should have the name <inliner-interactive-channel-base>.in, while the outgoing name should be <inliner-interactive-channel-base>.out">;
defm : BoolField<"inliner-interactive-include-default", "false",
  "In interactive mode, also send the default policy decision: inlining_default">;
defm : BoolField<"ml-advisor-keep-fpi-cache", "false",
  "For test - keep the ML Inline advisor's FunctionPropertiesInfo cache">;
defm : ValueField<"ml-advisor-size-increase-threshold", "float", "2.0",
  "Maximum factor by which expected native size may increase before blocking any further inlining">;
defm : ValueField<"ml-inliner-model-selector", "StringRef", "", "">;
defm : EnumField<"ml-inliner-skip-policy", "SkipMLPolicyCriteria", "SkipMLPolicyCriteria::Never", "never,if-caller-not-cold", ["SkipMLPolicyCriteria::Never", "SkipMLPolicyCriteria::IfCallerIsNotCold"], "">;
defm : BoolField<"ml-inliner-stop-immediately", "false", "">;

// MemoryDependenceAnalysis.cpp
defm : ValueField<"memdep-block-number-limit", "unsigned", "200",
  "The number of blocks to scan during memory dependency analysis">;
// Limit for the number of instructions to scan in a block.
defm : ValueField<"memdep-block-scan-limit", "unsigned", "100",
  "The number of instructions to scan in a block in memory dependency analysis">;
defm : ValueField<"memdep-cache-global-limit", "unsigned", "10000",
  "The max number of entries allowed in a cache">;

// MemorySSA.cpp
defm : ValueField<"dot-cfg-mssa", "StringRef", "",
  "file name for generated dot file">;
defm : ValueField<"memssa-check-limit", "unsigned", "100",
  "The maximum number of stores/phis MemorySSA will consider trying to walk past">;

// ModuleSummaryAnalysis.cpp
defm : BoolField<"enable-memprof-indirect-call-support", "true",
  "Enable MemProf support for summarizing and cloning indirect calls">;
// Force edges cold, which blocks importing when -import-cold-multiplier is 0.
// Useful for debugging.
defm : EnumField<"force-summary-edges-cold", "ForceSummaryHotness", "ForceSummaryHotness::None", "none,all-non-critical,all", ["ForceSummaryHotness::None", "ForceSummaryHotness::AllNonCritical", "ForceSummaryHotness::All"],
  "Force all edges in the function summary to cold">;
defm : ValueField<"module-summary-dot-file", "StringRef", "",
  "File to emit dot graph of new summary into">;
// This can be used to override the number of callees created from VP metadata
// normally taken from the -icp-max-prom option with a larger amount, if useful
// for analysis. Use a separate option so that we can control the number of
// indirect callees for ThinLTO summary based analysis (e.g. for MemProf which
// needs this information for a correct and not overly-conservative callsite
// graph analysis, especially because allocation contexts may not be very
// frequent), without affecting normal ICP.
defm : ValueField<"module-summary-max-indirect-edges", "unsigned", "0",
  "Max number of summary edges added from indirect call profile metadata">;

// PHITransAddr.cpp
defm : BoolField<"gvn-add-phi-translation", "false",
  "Enable phi-translation of add instructions">;

// ProfileSummaryInfo.cpp
defm : BoolField<"scale-partial-sample-profile-working-set-size", "true",
  "If true, scale the working set size of the partial sample profile by the partial profile ratio to reflect the size of the program being compiled">;
defm : BoolField<"partial-profile", "false",
  "Specify the current profile is used as a partial profile">;
defm : ValueField<"partial-sample-profile-working-set-size-scale-factor", "double", "0.008",
  "The scale factor used to scale the working set size of the partial sample profile along with the partial profile ratio. This includes the factor of the profile counter per block and the factor to scale the working set size to use the same shared thresholds as PGO">;

// RegionPrinter.cpp
defm : BoolField<"only-simple-regions", "false",
  "Show only simple regions in the graphviz viewer">;

// ScalarEvolution.cpp
defm : BoolField<"scalar-evolution-classify-expressions", "true",
  "When printing analysis, include information on every instruction">;
defm : BoolField<"scalar-evolution-finite-loop", "true",
  "Handle <= and >= in finite loops">;
defm : ValueField<"scalar-evolution-huge-expr-threshold", "unsigned", "4096",
  "Size of the expression which is considered huge">;
defm : ValueField<"scalar-evolution-max-add-rec-size", "unsigned", "8",
  "Max coefficients in AddRec during evolving">;
defm : ValueField<"scalar-evolution-max-arith-depth", "unsigned", "32",
  "Maximum depth of recursive arithmetics">;
defm : ValueField<"scalar-evolution-max-cast-depth", "unsigned", "8",
  "Maximum depth of recursive SExt/ZExt/Trunc">;
defm : ValueField<"scalar-evolution-max-constant-evolving-depth", "unsigned", "32",
  "Maximum depth of recursive constant evolving">;
defm : ValueField<"scalar-evolution-max-guard-dom-tree-steps", "unsigned", "8",
  "Maximum number of immediate dominator steps taken when collecting loop guards">;
defm : ValueField<"scalar-evolution-max-iterations", "unsigned", "100",
  "Maximum number of iterations SCEV will symbolically execute a constant derived loop">;
defm : ValueField<"scalar-evolution-max-loop-guard-collection-depth", "unsigned", "1",
  "Maximum depth for recursive loop guard collection">;
defm : ValueField<"scalar-evolution-max-scev-operations-implication-depth", "unsigned", "2",
  "Maximum depth of recursive SCEV operations implication analysis">;
defm : ValueField<"scalar-evolution-max-value-compare-depth", "unsigned", "2",
  "Maximum depth of recursive value complexity comparisons">;
defm : BoolField<"scalar-evolution-use-context-for-no-wrap-flag-strenghening", "true",
  "Infer nuw/nsw flags using context where suitable">;
defm : BoolField<"scalar-evolution-use-expensive-range-sharpening", "false",
  "Use more powerful methods of sharpening expression ranges. May be costly in terms of compile time">;
defm : ValueField<"scev-addops-inline-threshold", "unsigned", "500",
  "Threshold for inlining addition operands into a SCEV">;
defm : ValueField<"scev-mulops-inline-threshold", "unsigned", "32",
  "Threshold for inlining multiplication operands into a SCEV">;
defm : ValueField<"scev-range-iter-threshold", "unsigned", "32",
  "Threshold for switching to iteratively computing SCEV ranges">;
defm : BoolField<"scev-verify-ir", "false",
  "Verify IR correctness when making sensitive SCEV queries (slow)">;
defm : BoolField<"verify-scev-strict", "false",
  "Enable stricter verification with -verify-scev is passed">;

// StackSafetyAnalysis.cpp
defm : ValueField<"stack-safety-max-iterations", "int", "20", "">;
defm : BoolField<"stack-safety-print", "false", "">;
defm : BoolField<"stack-safety-run", "false", "">;

// StaticDataProfileInfo.cpp
// FIXME: This option is added for incremental rollout purposes.
// After the option, string literal partitioning should be implied by
// AnnotateStaticDataSectionPrefix in MemProfUse.cpp and this option should be
// cleaned up.
defm : BoolField<"memprof-annotate-string-literal-section-prefix", "true",
  "If true, annotate the string literal data section prefix">;
defm : BoolField<"preserve-hot-data-section-prefix", "true",
  "If true, hot data section prefixes are preserved">;

// VectorUtils.cpp
// Maximum factor for an interleaved memory access.
defm : ValueField<"max-interleave-group-factor", "unsigned", "8",
  "Maximum factor for an interleaved access group">;

// TODO: Remove the old spellings after 2027-04.
def : AliasOf<"da-disable-delinearization-checks", "da-delinearization-checks=false">;
def : AliasOf<"disable-gep-const-evaluation", "gep-const-evaluation=false">;
