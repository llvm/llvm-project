//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def ScalarOptions : OptionsStruct;

// ADCE.cpp
// This is a temporary option until we change the interface to this pass based
// on optimization level.
defm : BoolField<"adce-remove-control-flow", "true", "">;
// This option enables removing of may-be-infinite loops which have no other
// effect.
defm : BoolField<"adce-remove-loops", "false", "">;

// CallSiteSplitting.cpp
// Only allow instructions before a call, if their CodeSize cost is below
// DuplicationThreshold. Those instructions need to be duplicated in all split
// blocks.
defm : ValueField<"callsite-splitting-duplication-threshold", "unsigned", "5",
  "Only allow instructions before a call, if their cost is below DuplicationThreshold">;

// ConstantHoisting.cpp
defm : BoolField<"consthoist-gep", "false",
  "Try hoisting constant gep expressions">;
defm : ValueField<"consthoist-min-num-to-rebase", "unsigned", "0",
  "Do not rebase if number of dependent constants of a Base is less than this number">;
defm : BoolField<"consthoist-with-block-frequency", "true",
  "Enable the use of the block frequency analysis to reduce the chance to execute const materialization more frequently than without hoisting">;

// ConstraintElimination.cpp
defm : BoolField<"constraint-elimination-dump-reproducers", "false",
  "Dump IR to reproduce successful transformations">;
defm : ValueField<"constraint-elimination-max-rows", "unsigned", "500",
  "Maximum number of rows to keep in constraint system">;

// DeadStoreElimination.cpp
defm : ValueField<"dse-max-dom-cond-depth", "unsigned", "1024",
  "Max dominator tree recursion depth for eliminating redundant stores via dominating conditions">;
defm : ValueField<"dse-memoryssa-defs-per-block-limit", "unsigned", "5000",
  "The number of MemoryDefs we consider as candidates to eliminated other stores per basic block">;
defm : ValueField<"dse-memoryssa-otherbb-cost", "unsigned", "5",
  "The cost of a step in a different basic block than the killing MemoryDef">;
defm : ValueField<"dse-memoryssa-partial-store-limit", "unsigned", "5",
  "The maximum number candidates that only partially overwrite the killing MemoryDef to consider">;
defm : ValueField<"dse-memoryssa-path-check-limit", "unsigned", "50",
  "The maximum number of blocks to check when trying to prove that all paths to an exit go through a killing block">;
defm : ValueField<"dse-memoryssa-samebb-cost", "unsigned", "1",
  "The cost of a step in the same basic block as the killing MemoryDef">;
defm : ValueField<"dse-memoryssa-scanlimit", "unsigned", "150",
  "The number of memory instructions to scan for dead store elimination">;
defm : ValueField<"dse-memoryssa-walklimit", "unsigned", "90",
  "The maximum number of steps while walking upwards to find MemoryDefs that may be killed">;
// This flags allows or disallows DSE to optimize MemorySSA during its
// traversal. Note that DSE optimizing MemorySSA may impact other passes
// downstream of the DSE invocation and can lead to issues not being
// reproducible in isolation (i.e. when MemorySSA is built from scratch). In
// those cases, the flag can be used to check if DSE's MemorySSA optimizations
// impact follow-up passes.
defm : BoolField<"dse-optimize-memoryssa", "true",
  "Allow DSE to optimize memory accesses">;
// TODO: remove this flag.
defm : BoolField<"enable-dse-initializes-attr-improvement", "true",
  "Enable the initializes attr improvement in DSE">;
defm : BoolField<"enable-dse-partial-overwrite-tracking", "true",
  "Enable partial-overwrite tracking in DSE">;
defm : BoolField<"enable-dse-partial-store-merging", "true",
  "Enable partial store merging in DSE">;

// DFAJumpThreading.cpp
defm : ValueField<"dfa-cost-threshold", "unsigned", "50",
  "Maximum cost accepted for the transformation">;
defm : BoolField<"dfa-early-exit-heuristic", "true",
  "Exit early if an unpredictable value come from the same loop">;
defm : BoolField<"dfa-jump-view-cfg-before", "false",
  "View the CFG before DFA Jump Threading">;
defm : ValueField<"dfa-max-cloned-rate", "double", "7.5",
  "Maximum cloned instructions rate accepted for the transformation">;
defm : ValueField<"dfa-max-num-paths", "unsigned", "200",
  "Max number of paths enumerated around a switch">;
defm : ValueField<"dfa-max-num-visited-paths", "unsigned", "2500",
  "Max number of blocks visited while enumerating paths around a switch">;
defm : ValueField<"dfa-max-out-use-blocks", "unsigned", "40",
  "Maximum unduplicated blocks with outer uses accepted for the transformation">;
defm : ValueField<"dfa-max-path-length", "unsigned", "20",
  "Max number of blocks searched to find a threading path">;

// EarlyCSE.cpp
defm : BoolField<"earlycse-debug-hash", "false",
  "Perform extra assertion checking to verify that SimpleValue's hash function is well-behaved w.r.t. its isEqual predicate">;
defm : ValueField<"earlycse-mssa-optimization-cap", "unsigned", "500",
  "Enable imprecision in EarlyCSE in pathological cases, in exchange for faster compile. Caps the MemorySSA clobbering calls">;

// ExpandMemCmp.cpp
defm : ValueField<"max-loads-per-memcmp", "std::optional<unsigned>", "std::nullopt",
  "Set maximum number of loads used in expanded memcmp">;
defm : ValueField<"max-loads-per-memcmp-opt-size", "std::optional<unsigned>", "std::nullopt",
  "Set maximum number of loads used in expanded memcmp for -Os/Oz">;
defm : ValueField<"memcmp-num-loads-per-block", "std::optional<unsigned>", "std::nullopt",
  "The number of loads per basic block for inline expansion of memcmp">;

// Float2Int.cpp
// The largest integer type worth dealing with.
defm : ValueField<"float2int-max-integer-bw", "unsigned", "64",
  "Max integer bitwidth to consider in float2int">;

// GuardWidening.cpp
defm : BoolField<"guard-widening-widen-branch-guards", "true",
  "Whether or not we should widen guards  expressed as branches by widenable conditions">;

// GVN.cpp
defm : OptionalBoolField<"enable-gvn-memdep", "">;
defm : BoolField<"enable-gvn-memoryssa", "false", "">;
defm : BoolField<"enable-load-in-loop-pre", "true", "">;
defm : BoolField<"enable-load-pre", "true", "">;
defm : BoolField<"enable-scalar-pre", "true", "">;
defm : BoolField<"enable-split-backedge-in-load-pre", "false", "">;
// This is based on IsValueFullyAvailableInBlockNumSpeculationsMax stat.
defm : ValueField<"gvn-max-block-speculations", "uint32_t", "600",
  "Max number of blocks we're willing to speculate on (and recurse into) when deducing if a value is fully available or not in GVN">;
defm : ValueField<"gvn-max-num-deps", "uint32_t", "100",
  "Max number of dependences to attempt Load PRE">;
defm : ValueField<"gvn-max-num-insns", "uint32_t", "100",
  "Max number of instructions to scan in each basic block in GVN">;
defm : ValueField<"gvn-max-num-reaching-blocks", "uint32_t", "200",
  "Max number of blocks scanned per load in the MemorySSA reaching-value analysis">;
defm : ValueField<"gvn-max-num-visited-insts", "uint32_t", "100",
  "Max number of visited instructions when trying to find dominating value of select dependency">;
defm : ValueField<"gvn-scan-users-limit", "unsigned", "100",
  "The number of memory accesses to scan in a block in reaching memory values analysis">;

// GVNHoist.cpp
defm : ValueField<"gvn-hoist-max-bbs", "int", "4",
  "Max number of basic blocks on the path between hoisting locations (unlimited = -1)">;
defm : ValueField<"gvn-hoist-max-chain-length", "int", "10",
  "Maximum length of dependent chains to hoist (unlimited = -1)">;
defm : ValueField<"gvn-hoist-max-depth", "int", "100",
  "Hoist instructions from the beginning of the BB up to the maximum specified depth (unlimited = -1)">;
defm : ValueField<"gvn-max-hoisted", "int", "-1",
  "Max number of instructions to hoist (unlimited = -1)">;

// InductiveRangeCheckElimination.cpp
defm : BoolField<"irce-allow-narrow-latch", "true",
  "If set to true, IRCE may eliminate wide range checks in loops with narrow latch condition">;
defm : BoolField<"irce-allow-unsigned-latch", "true", "">;
defm : ValueField<"irce-loop-size-cutoff", "unsigned", "64", "">;
defm : ValueField<"irce-max-type-size-for-overflow-check", "unsigned", "32",
  "Maximum size of range check type for which can be produced runtime overflow check of its limit's computation">;
defm : ValueField<"irce-min-eliminated-checks", "unsigned", "10", "">;
defm : BoolField<"irce-print-changed-loops", "false", "">;
defm : BoolField<"irce-print-range-checks", "false", "">;
defm : BoolField<"irce-print-scaled-boundary-range-checks", "false", "">;
defm : BoolField<"irce-skip-profitability-checks", "false", "">;

// IndVarSimplify.cpp
defm : BoolField<"disable-lftr", "false",
  "Disable Linear Function Test Replace optimization">;
defm : BoolField<"indvars-post-increment-ranges", "true",
  "Use post increment control-dependent ranges in IndVarSimplify">;
defm : BoolField<"indvars-predicate-loop-traps", "true",
  "Predicate conditions that trap in loops with only local writes">;
defm : BoolField<"indvars-predicate-loops", "true",
  "Predicate conditions in read only loops">;
defm : BoolField<"indvars-widen-indvars", "true",
  "Allow widening of indvars to eliminate s/zext">;
defm : EnumField<"replexitval", "ReplaceExitVal", "OnlyCheapRepl",
  "never,cheap,unusedindvarinloop,noharduse,always",
  ["NeverRepl", "OnlyCheapRepl", "UnusedIndVarInLoop", "NoHardUse", "AlwaysRepl"],
  "Choose the strategy to replace exit value in IndVarSimplify">;

// JumpTableToSwitch.cpp
// TODO: Consider adding a cost model for profitability analysis of this
// transformation. Currently we replace a jump table with a switch if all the
// functions in the jump table are smaller than the provided threshold.
defm : ValueField<"jump-table-to-switch-function-size-threshold", "unsigned", "50",
  "Only split jump tables containing functions whose sizes are less or equal than this threshold">;
defm : ValueField<"jump-table-to-switch-size-threshold", "unsigned", "10",
  "Only split jump tables with size less or equal than JumpTableSizeThreshold">;

// JumpThreading.cpp
defm : BoolField<"jump-threading-across-loop-headers", "false",
  "Allow JumpThreading to thread across loop headers, for testing">;
defm : ValueField<"jump-threading-implication-search-threshold", "unsigned", "3",
  "The number of predecessors to search for a stronger condition to use to thread over a weaker condition">;
defm : ValueField<"jump-threading-phi-threshold", "unsigned", "76",
  "Max PHIs in BB to duplicate for jump threading">;
defm : ValueField<"jump-threading-threshold", "std::optional<unsigned>", "std::nullopt",
  "Max block size to duplicate for jump threading">;

// LICM.cpp
// Memory promotion is enabled by default.
defm : BoolField<"disable-licm-promotion", "false",
  "Disable memory promotion in LICM pass">;
defm : ValueField<"licm-max-num-fp-reassociations", "unsigned", "5U",
  "Set upper limit for the number of transformations performed during a single round of hoisting the reassociated expressions">;
defm : ValueField<"licm-max-num-int-reassociations", "unsigned", "5U",
  "Set upper limit for the number of transformations performed during a single round of hoisting the reassociated expressions">;
defm : ValueField<"licm-max-num-uses-traversed", "uint32_t", "8",
  "Max num uses visited for identifying load invariance in loop using invariant start">;
// Experimentally, memory promotion carries less importance than sinking and
// hoisting. Limit when we do promotion when using MemorySSA, in order to save
// compile time.
defm : ValueField<"licm-mssa-max-acc-promotion", "unsigned", "250",
  "[LICM & MemorySSA] When MSSA in LICM is disabled, this has no effect. When MSSA in LICM is enabled, then this is the maximum number of accesses allowed to be present in a loop in order to enable memory promotion">;
// Experimental option to allow imprecision in LICM in pathological cases, in
// exchange for faster compile. This is to be removed if MemorySSA starts to
// address the same issue. LICM calls MemorySSAWalker's
// getClobberingMemoryAccess, up to the value of the Cap, getting perfect
// accuracy. Afterwards, LICM will call into MemorySSA's getDefiningAccess,
// which may not be precise, since optimizeUses is capped. The result is
// correct, but we may not get as "far up" as possible to get which access is
// clobbering the one queried.
defm : ValueField<"licm-mssa-optimization-cap", "unsigned", "100",
  "Enable imprecision in LICM in pathological cases, in exchange for faster compile. Caps the MemorySSA clobbering calls">;

// LoopDataPrefetch.cpp
// By default, we limit this to creating 16 PHIs (which is a little over half of
// the allocatable register set).
defm : OptionalBoolField<"loop-prefetch-writes",
  "Prefetch write addresses">;
defm : ValueField<"max-prefetch-iters-ahead", "std::optional<unsigned>", "std::nullopt",
  "Max number of iterations to prefetch ahead">;
defm : ValueField<"min-prefetch-stride", "std::optional<unsigned>", "std::nullopt",
  "Min stride to add prefetches">;
defm : ValueField<"prefetch-distance", "std::optional<unsigned>", "std::nullopt",
  "Number of instructions to prefetch ahead">;

// LoopDeletion.cpp
defm : BoolField<"loop-deletion-enable-symbolic-execution", "true",
  "Break backedge through symbolic execution of 1st iteration attempting to prove that the backedge is never taken">;

// LoopDistribute.cpp
defm : BoolField<"enable-loop-distribute", "false",
  "Enable the new, experimental LoopDistribution Pass">;
defm : BoolField<"loop-distribute-non-if-convertible", "false",
  "Whether to distribute into a loop that may not be if-convertible by the loop vectorizer">;
defm : ValueField<"loop-distribute-scev-check-threshold", "unsigned", "8",
  "The maximum number of SCEV checks allowed for Loop Distribution">;
defm : ValueField<"loop-distribute-scev-check-threshold-with-pragma", "unsigned", "128",
  "The maximum number of SCEV checks allowed for Loop Distribution for loop marked with #pragma clang loop distribute(enable)">;
defm : BoolField<"loop-distribute-verify", "false",
  "Turn on DominatorTree and LoopInfo verification after Loop Distribution">;

// LoopFlatten.cpp
defm : BoolField<"loop-flatten-assume-no-overflow", "false",
  "Assume that the product of the two iteration trip counts will never overflow">;
defm : ValueField<"loop-flatten-cost-threshold", "unsigned", "2",
  "Limit on the cost of instructions that can be repeated due to loop flattening">;
defm : BoolField<"loop-flatten-version-loops", "true",
  "Version loops if flattened loop could overflow">;
defm : BoolField<"loop-flatten-widen-iv", "true",
  "Widen the loop induction variables, if possible, so overflow checks won't reject flattening">;

// LoopFuse.cpp
defm : ValueField<"loop-fusion-peel-max-count", "uint32_t", "0",
  "Max number of iterations to be peeled from a loop, such that fusion can take place">;

// LoopIdiomRecognize.cpp
defm : EnumField<"loop-idiom-crc-strategy", "CRCStrategyKind", "CRCStrategyKind::Auto",
  "disable,auto,table,clmul",
  ["CRCStrategyKind::Disable", "CRCStrategyKind::Auto", "CRCStrategyKind::Table", "CRCStrategyKind::Clmul"],
  "Preferred strategy for optimizing CRC loops">;
defm : BoolField<"loop-idiom-force-memset-pattern-intrinsic", "false",
  "Use memset.pattern intrinsic whenever possible">;
defm : BoolField<"use-lir-code-size-heurs", "true",
  "Use loop idiom recognition code size heuristics when compiling with -Os/-Oz">;

// LoopInterchange.cpp
// Maximum loop depth supported.
defm : ValueField<"loop-interchange-max-loop-nest-depth", "unsigned", "10",
  "Maximum depth of loop nest considered for the transform">;
defm : ValueField<"loop-interchange-max-mem-instr-ratio", "unsigned", "4",
  "Maximum number of load/store instructions squared in relation to the total number of instructions. Higher value may lead to more interchanges at the cost of compile-time">;
// Minimum loop depth supported.
defm : ValueField<"loop-interchange-min-loop-nest-depth", "unsigned", "2",
  "Minimum depth of loop nest considered for the transform">;
// Support for the inner-loop reduction pattern.
defm : BoolField<"loop-interchange-reduction-to-mem", "false",
  "Support for the inner-loop reduction pattern">;
defm : ValueField<"loop-interchange-threshold", "int", "0",
  "Interchange if you gain more than this number">;

// LoopLoadElimination.cpp
defm : ValueField<"loop-load-elimination-scev-check-threshold", "unsigned", "8",
  "The maximum number of SCEV checks allowed for Loop Load Elimination">;
defm : ValueField<"runtime-check-per-loop-load-elim", "unsigned", "1",
  "Max number of memchecks allowed per eliminated load on average">;

// LoopPredication.cpp
defm : BoolField<"loop-predication-enable-count-down-loop", "true", "">;
defm : BoolField<"loop-predication-enable-iv-truncation", "true", "">;
defm : BoolField<"loop-predication-insert-assumes-of-predicated-guards-conditions", "true",
  "Whether or not we should insert assumes of conditions of predicated guards">;
// This is the scale factor for the latch probability. We use this during
// profitability analysis to find other exiting blocks that have a much higher
// probability of exiting the loop instead of loop exiting via latch. This value
// should be greater than 1 for a sane profitability check.
defm : ValueField<"loop-predication-latch-probability-scale", "float", "2.0",
  "scale factor for the latch probability. Value should be greater than 1. Lower values are ignored">;
defm : BoolField<"loop-predication-predicate-widenable-branches-to-deopt", "true",
  "Whether or not we should predicate guards expressed as widenable branches to deoptimize blocks">;
defm : BoolField<"loop-predication-skip-profitability-checks", "false", "">;

// LoopRotation.cpp
// Experimentally allow loop header duplication. This should allow for better
// optimization at Oz, since loop-idiom recognition can then recognize things
// like memcpy. If this ends up being useful for many targets, we should drop
// this flag and make a code generation option that can be controlled
// independent of the opt level and exposed through the frontend.
defm : BoolField<"enable-loop-header-duplication-at-minsize", "false",
  "Enable loop header duplication even for minsize">;
defm : ValueField<"rotation-max-header-size", "unsigned", "16",
  "The default maximum header size for automatic loop rotation">;
defm : BoolField<"rotation-prepare-for-lto", "false",
  "Run loop-rotation in the prepare-for-lto stage. This option should be used for testing only">;

// LoopSimplifyCFG.cpp
defm : BoolField<"enable-loop-simplifycfg-term-folding", "true", "">;

// LoopSink.cpp
defm : ValueField<"max-uses-for-sinking", "unsigned", "30",
  "Do not sink instructions that have too many uses">;
defm : ValueField<"sink-freq-percent-threshold", "unsigned", "90",
  "Do not sink instructions that require cloning unless they execute less than this percent of the time">;

// LoopStrengthReduce.cpp
// Cleanup congruent phis after LSR phi expansion.
defm : BoolField<"enable-lsr-phielim", "true",
  "Enable LSR phi elimination">;
defm : ValueField<"lsr-complexity-limit", "unsigned", "std::numeric_limits<uint16_t>::max()",
  "LSR search space complexity limit">;
defm : BoolField<"lsr-drop-scaled-reg-for-vscale", "true",
  "Avoid using scaled registers with vscale-relative addressing">;
defm : OptionalBoolField<"lsr-drop-solution",
  "Attempt to drop solution if it is less profitable">;
defm : BoolField<"lsr-enable-vscale-immediates", "true",
  "Enable analysis of vscale-relative immediates in LSR">;
// Flag to choose how to narrow complex lsr solution.
defm : BoolField<"lsr-exp-narrow", "false",
  "Narrow LSR complex solution using expectation of registers number">;
// Flag to narrow search space by filtering non-optimal formulae with the same
// ScaledReg and Scale.
defm : BoolField<"lsr-filter-same-scaled-reg", "true",
  "Narrow LSR search space by filtering non-optimal formulae with the same ScaledReg and Scale">;
// The flag adds instruction count to solutions cost comparison.
defm : OptionalBoolField<"lsr-insns-cost",
  "Add instruction count to a LSR cost model">;
defm : EnumField<"lsr-preferred-addressing-mode", "std::optional<TargetTransformInfo::AddressingModeKind>", "std::nullopt",
  "none,preindexed,postindexed,all",
  ["TargetTransformInfo::AMK_None", "TargetTransformInfo::AMK_PreIndexed", "TargetTransformInfo::AMK_PostIndexed", "TargetTransformInfo::AMK_All"],
  "A flag that overrides the target's preferred addressing mode">;
defm : ValueField<"lsr-setupcost-depth-limit", "unsigned", "7",
  "The limit on recursion depth for LSRs setup cost">;

// LoopUnrollAndJamPass.cpp
defm : OptionalBoolField<"allow-unroll-and-jam",
  "Allows loops to be unroll-and-jammed">;
defm : ValueField<"pragma-unroll-and-jam-threshold", "unsigned", "1024",
  "Unrolled size limit for loops with an unroll_and_jam(full) or unroll_count pragma">;
defm : ValueField<"unroll-and-jam-count", "std::optional<unsigned>", "std::nullopt",
  "Use this unroll count for all loops including those with unroll_and_jam_count pragma values, for testing purposes">;
defm : ValueField<"unroll-and-jam-threshold", "std::optional<unsigned>", "std::nullopt",
  "Threshold to use for inner loop when doing unroll and jam">;

// LoopUnrollPass.cpp
defm : ValueField<"flat-loop-tripcount-threshold", "unsigned", "5",
  "If the runtime tripcount for the loop is lower than the threshold, the loop is considered as flat and will be less aggressively unrolled">;
defm : BoolField<"forget-scev-loop-unroll", "false",
  "Forget everything in SCEV when doing LoopUnroll, instead of just the current top-most loop. This is sometimes preferred to reduce compile time">;
defm : ValueField<"pragma-unroll-full-max-iterations", "unsigned", "1'000'000",
  "Maximum allowed iterations to unroll under pragma unroll full">;
defm : ValueField<"pragma-unroll-threshold", "unsigned", "16 * 1024",
  "Unrolled size limit for loops with unroll metadata (full, enable, or count)">;
defm : OptionalBoolField<"unroll-allow-partial",
  "Allows loops to be partially unrolled until -unroll-threshold loop size is reached">;
defm : OptionalBoolField<"unroll-allow-remainder",
  "Allow generation of a loop remainder (extra iterations) when unrolling a loop">;
defm : ValueField<"unroll-count", "std::optional<unsigned>", "std::nullopt",
  "Use this unroll count for all loops including those with unroll_count pragma values, for testing purposes">;
defm : ValueField<"unroll-full-max-count", "std::optional<unsigned>", "std::nullopt",
  "Set the max unroll count for full unrolling, for testing purposes">;
defm : ValueField<"unroll-max-count", "std::optional<unsigned>", "std::nullopt",
  "Set the max unroll count for partial and runtime unrolling, fortesting purposes">;
defm : ValueField<"unroll-max-iteration-count-to-analyze", "std::optional<unsigned>", "std::nullopt",
  "Don't allow loop unrolling to simulate more than this number of iterations when checking full unroll profitability">;
defm : ValueField<"unroll-max-percent-threshold-boost", "std::optional<unsigned>", "std::nullopt",
  "The maximum 'boost' (represented as a percentage >= 100) applied to the threshold when aggressively unrolling a loop due to the dynamic cost savings. If completely unrolling a loop will reduce the total runtime from X to Y, we boost the loop unroll threshold to DefaultThreshold*std::min(MaxPercentThresholdBoost, X/Y). This limit avoids excessive code bloat">;
defm : ValueField<"unroll-max-upperbound", "std::optional<unsigned>", "std::nullopt",
  "The max of trip count upper bound that is considered in unrolling">;
defm : ValueField<"unroll-optsize-threshold", "unsigned", "0",
  "The cost threshold for loop unrolling when optimizing for size">;
defm : ValueField<"unroll-partial-threshold", "std::optional<unsigned>", "std::nullopt",
  "The cost threshold for partial loop unrolling">;
defm : OptionalBoolField<"unroll-remainder",
  "Allow the loop remainder to be unrolled">;
// This option isn't ever intended to be enabled, it serves to allow experiments
// to check the assumptions about when this kind of revisit is necessary.
defm : BoolField<"unroll-revisit-child-loops", "false",
  "Enqueue and re-visit child loops in the loop PM after unrolling. This shouldn't typically be needed as child loops (or their clones) were already visited">;
defm : OptionalBoolField<"unroll-runtime",
  "Unroll loops with run-time trip counts">;
defm : ValueField<"unroll-threshold", "std::optional<unsigned>", "std::nullopt",
  "The cost threshold for loop unrolling">;
defm : ValueField<"unroll-threshold-aggressive", "unsigned", "300",
  "Threshold (max size of unrolled loop) to use in aggressive (O3) optimizations">;
defm : ValueField<"unroll-threshold-default", "unsigned", "150",
  "Default threshold (max size of unrolled loop), used in all but O3 optimizations">;

// LoopVersioningLICM.cpp
// Threshold minimum allowed percentage for possible invariant instructions in a
// loop.
defm : ValueField<"licm-versioning-invariant-threshold", "float", "25",
  "LoopVersioningLICM's minimum allowed percentage of possible invariant instructions per loop">;
// Threshold for maximum allowed loop nest/depth.
defm : ValueField<"licm-versioning-max-depth-threshold", "unsigned", "2",
  "LoopVersioningLICM's threshold for maximum allowed loop nest/depth">;

// LowerExpectIntrinsic.cpp
// WARNING: these values are internal implementation detail of the pass. They
// should not be exposed to the outside of the pass, front-end codegen should
// emit @llvm.expect intrinsics instead of using these weights directly.
// Transforms should use TargetTransformInfo's getPredictableBranchThreshold().
defm : ValueField<"likely-branch-weight", "uint32_t", "2000",
  "Weight of the branch likely to be taken">;
defm : ValueField<"unlikely-branch-weight", "uint32_t", "1",
  "Weight of the branch unlikely to be taken">;

// LowerMatrixIntrinsics.cpp
defm : BoolField<"force-fuse-matrix", "false",
  "Force matrix instruction fusion even if not profitable">;
defm : BoolField<"fuse-matrix", "true",
  "Enable/disable fusing matrix instructions">;
defm : ValueField<"fuse-matrix-loops-threshold", "unsigned", "200",
  "Generate loop nests for tiling when expected number of operations exceeds threshold">;
// TODO: Allow and use non-square tiles.
defm : ValueField<"fuse-matrix-tile-size", "unsigned", "4",
  "Tile size for matrix instruction fusion using square-shaped tiles">;
defm : BoolField<"matrix-allow-contract", "false",
  "Allow the use of FMAs if available and profitable. This may result in different results, due to less rounding error">;
defm : EnumField<"matrix-default-layout", "MatrixLayoutTy", "MatrixLayoutTy::ColumnMajor",
  "column-major,row-major",
  ["MatrixLayoutTy::ColumnMajor", "MatrixLayoutTy::RowMajor"],
  "Sets the default matrix layout">;
defm : BoolField<"matrix-print-after-transpose-opt", "false", "">;
defm : ValueField<"matrix-split-matmul-remainder-over-threshold", "unsigned", "0",
  "Illegal remainder vectors over this size in bits should be split in the inner loop of matmul">;
defm : BoolField<"verify-matrix-shapes", "false",
  "Enable/disable matrix shape verification">;

// NewGVN.cpp
// Currently, the generation "phi of ops" can result in correctness issues.
defm : BoolField<"enable-phi-of-ops", "true", "">;
// Currently store defining access refinement is too slow due to basicaa being
// egregiously slow. This flag lets us keep it working while we work on this
// issue.
defm : BoolField<"enable-store-refinement", "false", "">;

// PlaceSafepoints.cpp
// Ignore opportunities to avoid placing safepoints on backedges, useful for
// validation.
defm : BoolField<"spp-all-backedges", "false", "">;
// How narrow does the trip count of a loop have to be to have to be considered
// "counted"? Counted loops do not get safepoints at backedges.
defm : ValueField<"spp-counted-loop-trip-width", "int", "32", "">;
defm : BoolField<"spp-no-backedge", "false", "">;
defm : BoolField<"spp-no-call", "false", "">;
defm : BoolField<"spp-no-entry", "false", "">;
// If true, split the backedge of a loop when placing the safepoint, otherwise
// split the latch block itself. Both are useful to support for experimentation,
// but in practice, it looks like splitting the backedge optimizes better.
defm : BoolField<"spp-split-backedge", "false", "">;

// Reassociate.cpp
defm : BoolField<"reassociate-use-cse-local", "true",
  "Only reorder expressions within a basic block when exposing CSE opportunities">;

// RewriteStatepointsForGC.cpp
defm : BoolField<"rs4gc-allow-statepoint-with-no-deopt-info", "true", "">;
defm : BoolField<"rs4gc-remat-derived-at-uses", "true", "">;
// Print out the base pointers for debugging.
defm : BoolField<"spp-print-base-pointers", "false", "">;
// Print the liveset found at the insert location.
defm : BoolField<"spp-print-liveset", "false", "">;
defm : BoolField<"spp-print-liveset-size", "false", "">;
// Cost threshold measuring when it is profitable to rematerialize value instead
// of relocating it.
defm : ValueField<"spp-rematerialization-threshold", "unsigned", "6", "">;

// SeparateConstOffsetFromGEP.cpp
defm : BoolField<"disable-separate-const-offset-from-gep", "false",
  "Do not separate the constant offset from a GEP instruction">;
// Setting this flag may emit false positives when the input module already
// contains dead instructions. Therefore, we set it only in unit tests that are
// free of dead code.
defm : BoolField<"reassociate-geps-verify-no-dead-code", "false",
  "Verify this pass produces no dead code">;

// SimpleLoopUnswitch.cpp
defm : BoolField<"enable-nontrivial-unswitch", "false",
  "Forcibly enables non-trivial loop unswitching rather than following the configuration passed into the pass">;
defm : BoolField<"enable-unswitch-cost-multiplier", "true",
  "Enable unswitch cost multiplier that prohibits exponential explosion in nontrivial unswitch">;
defm : BoolField<"freeze-loop-unswitch-cond", "true",
  "If enabled, the freeze instruction will be added to condition of loop unswitch to prevent miscompilation">;
defm : BoolField<"simple-loop-unswitch-drop-non-trivial-implicit-null-checks", "false",
  "If enabled, drop make.implicit metadata in unswitched implicit null checks to save time analyzing if we can keep it">;
defm : BoolField<"simple-loop-unswitch-estimate-profile", "true", "">;
defm : BoolField<"simple-loop-unswitch-guards", "true",
  "If enabled, simple loop unswitching will also consider llvm.experimental.guard intrinsics as unswitch candidates">;
defm : ValueField<"simple-loop-unswitch-inject-invariant-condition-hotness-threshold", "unsigned", "16",
  "Only try to inject loop invariant conditions and unswitch on them to eliminate branches that are not-taken 1/<this option> times or less">;
defm : BoolField<"simple-loop-unswitch-inject-invariant-conditions", "true",
  "Whether we should inject new invariants and unswitch them to eliminate some existing (non-invariant) conditions">;
defm : ValueField<"simple-loop-unswitch-memoryssa-threshold", "unsigned", "100",
  "Max number of memory uses to explore during partial unswitching analysis">;
defm : ValueField<"unswitch-num-initial-unscaled-candidates", "int", "8",
  "Number of unswitch candidates that are ignored when calculating cost multiplier">;
defm : ValueField<"unswitch-parent-blocks-div", "int", "8",
  "Outer loop size divisor for cost multiplier">;
defm : ValueField<"unswitch-siblings-toplevel-div", "int", "2",
  "Toplevel siblings divisor for cost multiplier">;
defm : ValueField<"unswitch-threshold", "int", "50",
  "The cost threshold for unswitching a loop">;

// SimplifyCFGPass.cpp
defm : ValueField<"bonus-inst-threshold", "std::optional<unsigned>", "std::nullopt",
  "Control the number of bonus instructions">;
defm : OptionalBoolField<"forward-switch-cond",
  "Forward switch condition to phi ops">;
defm : OptionalBoolField<"hoist-common-insts",
  "hoist common instructions">;
defm : OptionalBoolField<"hoist-loads-stores-with-cond-faulting",
  "Hoist loads/stores if the target supports conditional faulting">;
defm : OptionalBoolField<"keep-loops",
  "Preserve canonical loop structure">;
defm : OptionalBoolField<"sink-common-insts",
  "Sink common instructions">;
defm : OptionalBoolField<"speculate-unpredictables",
  "Speculate unpredictable branches">;
defm : OptionalBoolField<"switch-range-to-icmp",
  "Convert switches into an integer range comparison">;
defm : OptionalBoolField<"switch-to-lookup",
  "Convert switches to lookup tables">;

// SpeculativeExecution.cpp
// Speculating just a few instructions from a larger block tends not to be
// profitable and this limit prevents that. A reason for that is that small
// basic blocks are more likely to be candidates for further optimization.
defm : ValueField<"spec-exec-max-not-hoisted", "unsigned", "5",
  "Speculative execution is not applied to basic blocks where the number of instructions that would not be speculatively executed exceeds this limit">;
// The risk that speculation will not pay off increases with the number of
// instructions speculated, so we put a limit on that.
defm : ValueField<"spec-exec-max-speculation-cost", "unsigned", "7",
  "Speculative execution is not applied to basic blocks where the cost of the instructions to speculatively execute exceeds this limit">;
defm : BoolField<"spec-exec-only-if-divergent-target", "false",
  "Speculative execution is applied only to targets with divergent branches, even if the pass was configured to apply only to all targets">;

// SROA.cpp
// Disable running mem2reg during SROA in order to test or debug SROA.
defm : BoolField<"sroa-skip-mem2reg", "false", "">;

// StraightLineStrengthReduce.cpp
// Only for testing.
defm : BoolField<"enable-poison-reuse-guard", "true",
  "Enable poison-reuse guard">;

// StructurizeCFG.cpp
defm : BoolField<"structurizecfg-relaxed-uniform-regions", "true",
  "Allow relaxed uniform region checks">;
defm : OptionalBoolField<"structurizecfg-skip-uniform-regions",
  "Force whether the StructurizeCFG pass skips uniform regions">;

// TailRecursionElimination.cpp
defm : BoolField<"disable-tail-call-elim-for-cold-calls", "false",
  "Disable tail call elimination and optimization for cold calls or in cold functions">;
defm : BoolField<"tre-disable-entrycount-recompute", "false",
  "Force disabling recomputing of function entry count, on successful tail recursion elimination">;
