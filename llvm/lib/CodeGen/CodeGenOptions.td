//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def CodeGenOptions : OptionsStruct;

// AggressiveAntiDepBreaker.cpp
let Condition = AssertsOnly in {
defm : ValueField<"agg-antidep-debugdiv", "int", "0",
  "Debug control for aggressive anti-dep breaker">;
defm : ValueField<"agg-antidep-debugmod", "int", "0",
  "Debug control for aggressive anti-dep breaker">;
}

// BasicBlockMatchingAndInference.cpp
defm : ValueField<"propeller-infer-threshold", "float", "0.6f",
  "Threshold for infer stale profile">;
defm : BoolField<"propeller-match-bbid", "false",
  "Match basic blocks using BBIDs instead of hashes for Propeller inference">;
let MetaVarName = "<filename>" in
defm : ValueField<"sample-profile-supplement", "StringRef", "",
  "Sample profile used to supplement the weights of basic blocks without samples in another profile. "
  "Currently used with basic-block-sections profile only.">;
let MetaVarName = "<filename>" in
defm : ValueField<"sample-profile-supplement-remapping-file", "StringRef", "",
  "Symbol remapping file for -sample-profile-supplement">;

// BranchFolding.cpp
defm : OptionalBoolField<"branch-folder-hoist-common-code",
  "Override common-code hoisting in the BranchFolding pass">;
defm : OptionalBoolField<"branch-folder-reorder-blocks",
  "Override basic-block reordering in the BranchFolding pass">;
defm : OptionalBoolField<"tail-merge", "">;
// Heuristic for tail merging (and, inversely, tail duplication).
defm : OptionalValueField<"tail-merge-size", "unsigned",
  "Min number of instructions to consider tail merging">;
// Throttle for huge numbers of predecessors (compile speed problems)
defm : ValueField<"tail-merge-threshold", "unsigned", "150",
  "Max number of predecessors to consider tail merging">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"enable-tail-merge", "tail-merge">;

// CodeGenPrepare.cpp
defm : BoolField<"bbsections-guided-section-prefix", "true",
  "Use the basic-block-sections profile to determine the text section prefix for hot functions. "
  "Functions with basic-block-sections profile will be placed in `.text.hot` regardless of their FDO profile info. "
  "Other functions won't be impacted, i.e., their prefixes will be decided by FDO/sampleFDO profiles">;
defm : BoolField<"cgp-addr-sink-combine-base-gv", "true",
  "Allow combining of BaseGV field in Address sinking">;
defm : BoolField<"cgp-addr-sink-combine-base-offs", "true",
  "Allow combining of BaseOffs field in Address sinking">;
defm : BoolField<"cgp-addr-sink-combine-base-reg", "true",
  "Allow combining of BaseReg field in Address sinking">;
defm : BoolField<"cgp-addr-sink-combine-scaled-reg", "true",
  "Allow combining of ScaledReg field in Address sinking">;
defm : BoolField<"cgp-addr-sink-new-phis", "false",
  "Allow creation of Phis in Address sinking">;
defm : BoolField<"cgp-addr-sink-new-select", "true",
  "Allow creation of selects in Address sinking">;
defm : BoolField<"cgp-andcmp-sinking", "true",
  "Enable sinking and/cmp into branches">;
defm : BoolField<"cgp-branch-opts", "true",
  "Enable branch optimizations in CodeGenPrepare">;
defm : BoolField<"cgp-complex-addr-modes", "true",
  "Enable combining addressing modes with different parts in optimizeMemoryInst">;
defm : BoolField<"cgp-delete-phis", "true",
  "Enable elimination of dead PHI nodes">;
defm : BoolField<"cgp-ext-ld-promotion", "true",
  "Enable ext(promotable(ld)) -> promoted(ext(ld)) optimization in CodeGenPrepare">;
defm : BoolField<"cgp-force-split-store", "false",
  "Force store splitting no matter what the target query says">;
defm : ValueField<"cgp-freq-ratio-to-skip-merge", "uint64_t", "2",
  "Skip merging empty blocks if (frequency of empty block) / (frequency of destination block) is greater than this ratio">;
defm : BoolField<"cgp-gc-opts", "true",
  "Enable GC optimizations in CodeGenPrepare">;
defm : ValueField<"cgp-huge-func", "unsigned", "10000",
  "Least BB number of huge function">;
defm : BoolField<"cgp-icmp-eq2icmp-st", "false",
  "Enable ICMP_EQ to ICMP_S(L|G)T conversion">;
defm : ValueField<"cgp-max-address-users-to-scan", "unsigned", "100",
  "Max number of address users to look at">;
defm : BoolField<"cgp-optimize-phi-types", "true",
  "Enable converting phi types in CodeGenPrepare">;
defm : BoolField<"cgp-preheader-prot", "true",
  "Protect loop preheaders from removal">;
defm : BoolField<"cgp-select2branch", "true",
  "Enable select to branch conversion">;
defm : BoolField<"cgp-split-large-offset-gep", "true",
  "Enable splitting large offset of GEP">;
defm : BoolField<"cgp-store-extract", "true",
  "Enable store(extract) optimizations in CodeGenPrepare">;
defm : BoolField<"cgp-stress-ext-ld-promotion", "false",
  "Stress test ext(promotable(ld)) -> promoted(ext(ld)) optimization in CodeGenPrepare">;
defm : BoolField<"cgp-stress-store-extract", "false",
  "Stress test store(extract) optimizations in CodeGenPrepare">;
defm : BoolField<"cgp-type-promotion-merge", "true",
  "Enable merging of redundant sexts when one is dominating the other">;
defm : BoolField<"cgp-verify-bfi-updates", "false",
  "Enable BFI update verification for CodeGenPrepare">;
defm : BoolField<"profile-guided-section-prefix", "true",
  "Use profile info to add section prefix for hot/cold functions">;
defm : BoolField<"profile-unknown-in-special-section", "false",
  "In profiling mode like sampleFDO, if a function doesn't have profile, we cannot tell the function is cold for sure because it may be a function newly added without ever being sampled. "
  "With the flag enabled, compiler can put such profile unknown functions into a special section, so runtime system can choose to handle it in a different way than .text section, to save RAM for example">;

// DFAPacketizer.cpp
defm : OptionalValueField<"dfa-instr-limit", "unsigned",
  "If present, stops packetizing after N instructions">;

// EarlyIfConversion.cpp
defm : ValueField<"early-ifcvt-cold-edge-threshold", "unsigned", "1",
  "Cold edge probability threshold in percent for using no critical-path extension budget (0 disables the threshold)">;
// Absolute maximum number of instructions allowed per speculated block.
// This bypasses all other heuristics, so it should be set fairly high.
defm : ValueField<"early-ifcvt-limit", "unsigned", "30",
  "Maximum number of instructions per speculated block">;
defm : ValueField<"early-ifcvt-max-region-instrs", "unsigned", "64",
  "Limit the number of blocks and instructions examined when searching for calls between a load and the condition it feeds">;
defm : ValueField<"early-ifcvt-max-steps", "unsigned", "16",
  "Limit the number of steps taken when searching for a recently loaded value">;
// Enable analysis of data dependent branches (conditions derived from loads).
defm : BoolField<"enable-early-ifcvt-data-dependent", "false",
  "Enable hard-to-predict branch analysis for if-conversion">;
// Stress testing mode - disable heuristics.
defm : BoolField<"stress-early-ifcvt", "false",
  "Turn all knobs to 11">;

// ExpandIRInsts.cpp
defm : OptionalValueField<"expand-div-rem-bits", "unsigned",
  "div and rem instructions on integers with more than <N> bits are expanded">;
defm : OptionalValueField<"expand-fp-convert-bits", "unsigned",
  "fp convert instructions on integers with more than <N> bits are expanded">;

// FixupStatepointCallerSaved.cpp
defm : BoolField<"fixup-allow-gcptr-in-csr", "false",
  "Allow passing GC Pointer arguments in callee saved registers">;
// This is purely debugging option.
// It may be handy for investigating statepoint spilling issues.
defm : OptionalValueField<"fixup-max-csr-statepoints", "unsigned",
  "Max number of statepoints allowed to pass GC Ptrs in registers">;
defm : BoolField<"fixup-scs-enable-copy-propagation", "true",
  "Enable simple copy propagation during register reloading">;
defm : BoolField<"fixup-scs-extend-slot-size", "false",
  "Allow spill in spill slot of greater size than register size">;

// HardwareLoops.cpp
defm : OptionalBoolField<"force-hardware-loop-guard",
  "Force generation of loop guard intrinsic">;
defm : OptionalBoolField<"force-hardware-loop-phi",
  "Force hardware loop counter to be updated through a phi">;
defm : OptionalBoolField<"force-hardware-loops",
  "Force hardware loops intrinsics to be inserted">;
defm : OptionalBoolField<"force-nested-hardware-loop",
  "Force allowance of nested hardware loops">;
defm : OptionalValueField<"hardware-loop-counter-bitwidth", "unsigned",
  "Set the loop counter bitwidth">;
defm : OptionalValueField<"hardware-loop-decrement", "unsigned",
  "Set the loop decrement value">;

// IfConversion.cpp
defm : BoolField<"ifcvt-branch-fold", "true", "">;
defm : BoolField<"ifcvt-diamond", "true", "">;
defm : ValueField<"ifcvt-fn-start", "int", "-1", "">;
defm : ValueField<"ifcvt-fn-stop", "int", "-1", "">;
defm : BoolField<"ifcvt-forked-diamond", "true", "">;
defm : ValueField<"ifcvt-limit", "int", "-1", "">;
defm : BoolField<"ifcvt-simple", "true", "">;
defm : BoolField<"ifcvt-simple-false", "true", "">;
defm : BoolField<"ifcvt-triangle", "true", "">;
defm : BoolField<"ifcvt-triangle-false", "true", "">;
defm : BoolField<"ifcvt-triangle-rev", "true", "">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-ifcvt-diamond", "ifcvt-diamond=false">;
def : AliasOf<"disable-ifcvt-triangle-false", "ifcvt-triangle-false=false">;

// ImplicitNullChecks.cpp
defm : ValueField<"imp-null-check-page-size", "int", "4096",
  "The page size of the target in bytes">;
defm : ValueField<"imp-null-max-insts-to-consider", "unsigned", "8",
  "The max number of instructions to consider hoisting loads over (the algorithm is quadratic over this number)">;

// MIRCanonicalizerPass.cpp
let MetaVarName = "<N>" in
defm : ValueField<"mir-canonicalizer-nth-function", "unsigned", "~0u",
  "Function number to canonicalize">;

// MIRVRegNamerUtils.cpp
defm : BoolField<"mir-vreg-namer-use-stable-hash", "false",
  "Use Stable Hashing for MIR VReg Renaming">;

// MachineCSE.cpp
defm : BoolField<"machine-cse-aggressive", "false",
  "Override the profitability heuristics for Machine CSE">;
// Threshold to avoid excessive cost to compute isProfitableToCSE.
defm : ValueField<"machine-cse-csuses-threshold", "int", "1024",
  "Threshold for the size of CSUses">;

// MachineCombiner.cpp
defm : BoolField<"machine-combiner-dump-subst-intrs", "false",
  "Dump all substituted intrs">;
defm : ValueField<"machine-combiner-inc-threshold", "unsigned", "500",
  "Incremental depth computation will be used for basic blocks with more instructions">;
defm : BoolField<"machine-combiner-verify-pattern-order", "MachineCombinerVerifyPatternOrderDefault",
  "Verify that the generated patterns are ordered by increasing latency">;

// MachineCopyPropagation.cpp
defm : OptionalBoolField<"mcp-spill-copy-elim", "">;
defm : BoolField<"mcp-use-is-copy-instr", "false", "">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"enable-spill-copy-elim", "mcp-spill-copy-elim">;

// MachineLICM.cpp
defm : BoolField<"machine-licm-avoid-speculation", "true",
  "MachineLICM should avoid speculation">;
// The default threshold of 100 (i.e. if target block is 100 times hotter)
// is based on empirical data on a single target and is subject to tuning.
defm : ValueField<"machine-licm-block-freq-ratio-threshold", "unsigned", "100",
  "Do not hoist instructions if target block is N times hotter than the source">;
defm : BoolField<"machine-licm-hoist-cheap-insts", "false",
  "MachineLICM should hoist even cheap instructions">;
defm : BoolField<"machine-licm-hoist-const-loads", "true",
  "Hoist invariant loads">;
defm : BoolField<"machine-licm-hoist-const-stores", "true",
  "Hoist invariant stores">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"block-freq-ratio-threshold", "machine-licm-block-freq-ratio-threshold">;
def : AliasOf<"hoist-const-loads", "machine-licm-hoist-const-loads">;
def : AliasOf<"hoist-const-stores", "machine-licm-hoist-const-stores">;

// MachineOutliner.cpp
defm : BoolField<"append-content-hash-outlined-name", "true",
  "This appends the content hash to the globally outlined function name. It's beneficial for enhancing the precision of the stable hash and for ordering the outlined functions">;
// Set to true if the user wants the outliner to run on linkonceodr linkage
// functions. This is false by default because the linker can dedupe linkonceodr
// functions. Since the outliner is confined to a single module (modulo LTO),
// this is off by default. It should, however, be the default behaviour in LTO.
defm : BoolField<"enable-linkonceodr-outlining", "false",
  "Enable the machine outliner on linkonceodr functions">;
defm : BoolField<"global-outlining", "true",
  "Enable global outlining, which generates or uses the codegen data">;
defm : BoolField<"machine-outliner-publish-unprofitable-candidates", "false",
  "Publish locally unprofitable candidates to Machine Outliner CGData">;
// Number of times to re-run the outliner. This is not the total number of runs
// as the outliner will run at least one time. The default value is set to 0,
// meaning the outliner will run one time and rerun zero times after that.
defm : ValueField<"machine-outliner-reruns", "unsigned", "0",
  "Number of times to rerun the outliner after the initial outline">;
// Chosen experimentally: longer sequences are rarely locally unprofitable, so
// publishing them mostly adds build time.
defm : ValueField<"machine-outliner-unprofitable-candidate-max-instrs", "unsigned", "3",
  "Maximum instruction count for locally unprofitable candidates">;
defm : ValueField<"outliner-benefit-threshold", "unsigned", "1",
  "The minimum size in bytes before an outlining candidate is accepted">;
defm : BoolField<"outliner-leaf-descendants", "true",
  "Consider all leaf descendants of internal nodes of the suffix tree as candidates for outlining (if false, only leaf children are considered)">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"disable-global-outlining", "global-outlining=false">;

// MachinePipeliner.cpp
defm : BoolField<"pipeliner-annotate-for-testing", "false",
  "Instead of emitting the pipelined code, annotate instructions with the generated schedule for feeding into the -modulo-schedule-test pass">;
let Condition = AssertsOnly in
defm : BoolField<"pipeliner-dbg-res", "false", "">;
// A command line option to enable the CopyToPhi DAG mutation.
defm : BoolField<"pipeliner-enable-copytophi", "true",
  "Enable CopyToPhi DAG Mutation">;
defm : BoolField<"pipeliner-experimental-cg", "false",
  "Use the experimental peeling code generator for software pipelining">;
// A command line argument to force pipeliner to use specified initial interval.
defm : ValueField<"pipeliner-force-ii", "int", "-1",
  "Force pipeliner to use specified II">;
// A command line argument to force pipeliner to use specified issue width.
defm : ValueField<"pipeliner-force-issue-width", "int", "-1",
  "Force pipeliner to use specified issue width">;
defm : BoolField<"pipeliner-ignore-recmii", "false",
  "Ignore RecMII">;
defm : ValueField<"pipeliner-ii-search-range", "int", "10",
  "Range to search for II">;
let Condition = AssertsOnly in
defm : ValueField<"pipeliner-max", "int", "-1", "">;
// A command line argument to limit minimum initial interval for pipelining.
defm : ValueField<"pipeliner-max-mii", "int", "27",
  "Size limit for the MII">;
// A command line argument to limit the number of store instructions in the
// target basic block.
defm : ValueField<"pipeliner-max-num-stores", "unsigned", "200",
  "Maximum number of stores allwed in the target loop">;
// A command line argument to limit the number of stages in the pipeline.
defm : ValueField<"pipeliner-max-stages", "int", "3",
  "Maximum stages allowed in the generated scheduled">;
defm : BoolField<"pipeliner-mve-cg", "false",
  "Use the MVE code generator for software pipelining">;
defm : BoolField<"pipeliner-opt-size", "false",
  "Enable SWP at Os">;
// A command line option to disable the pruning of chain dependences due to an
// unrelated Phi.
defm : BoolField<"pipeliner-prune-deps", "true",
  "Prune dependences between unrelated Phi nodes">;
defm : OptionalBoolField<"pipeliner-register-pressure",
  "Limit register pressure of scheduled loop">;
defm : ValueField<"pipeliner-register-pressure-margin", "int", "5",
  "Margin representing the unused percentage of the register pressure limit">;
let Condition = AssertsOnly in
defm : BoolField<"pipeliner-show-mask", "false", "">;
// A command line argument to set the window scheduling option.
defm : EnumField<"window-sched", "WindowSchedulingFlag", "", "off,on,force",
  ["WindowSchedulingFlag::Off", "WindowSchedulingFlag::On", "WindowSchedulingFlag::Force"],
  "Set how to use window scheduling algorithm">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"enable-pipeliner-opt-size", "pipeliner-opt-size">;

// MachineScheduler.cpp
defm : OptionalBoolField<"enable-misched",
  "Enable the machine instruction scheduling pass">;
defm : OptionalBoolField<"enable-post-misched",
  "Enable the post-ra machine instruction scheduling pass">;
defm : OptionalBoolField<"enable-ssa-misched",
  "Enable the machine instruction scheduling pass in SSA">;
defm : BoolField<"misched-cluster", "true",
  "Enable memop clustering">;
let Condition = AssertsOnly in
defm : ValueField<"misched-cutoff", "unsigned", "~0U",
  "Stop scheduling after N instructions">;
defm : BoolField<"misched-cyclicpath", "true",
  "Enable cyclic critical path analysis">;
defm : BoolField<"misched-dcpl", "false",
  "Print critical path length to stdout">;
let Condition = AssertsOnly in {
defm : BoolField<"misched-detail-resource-booking", "false",
  "Show details of invoking getNextResoufceCycle">;
defm : BoolField<"misched-dump-reserved-cycles", "false",
  "Dump resource usage at schedule boundary">;
defm : BoolField<"misched-dump-schedule-trace", "false",
  "Dump resource usage at schedule boundary">;
defm : ValueField<"misched-dump-schedule-trace-col-header-width", "unsigned", "19",
  "Set width of the columns with the resources and schedule units">;
defm : ValueField<"misched-dump-schedule-trace-col-width", "unsigned", "5",
  "Set width of the columns showing resource booking">;
}
defm : ValueField<"misched-fast-cluster-threshold", "unsigned", "1000",
  "The threshold for fast cluster">;
defm : BoolField<"misched-force-fast-cluster", "false",
  "Switch to fast cluster algorithm with the lost of some fusion opportunities">;
// Avoid quadratic complexity in unusually large basic blocks by limiting the
// size of the ready lists.
defm : ValueField<"misched-limit", "unsigned", "256",
  "Limit ready list to N instructions">;
let Condition = AssertsOnly in {
defm : OptionalValueField<"misched-only-block", "unsigned",
  "Only schedule this MBB#">;
defm : OptionalValueField<"misched-only-func", "StringRef",
  "Only schedule this function">;
}
defm : EnumField<"misched-postra-direction", "MISched::Direction", "",
  "topdown,bottomup,bidirectional",
  ["MISched::TopDown", "MISched::BottomUp", "MISched::Bidirectional"],
  "Post reg-alloc list scheduling direction">;
defm : EnumField<"misched-prera-direction", "MISched::Direction", "",
  "topdown,bottomup,bidirectional",
  ["MISched::TopDown", "MISched::BottomUp", "MISched::Bidirectional"],
  "Pre reg-alloc list scheduling direction">;
let Condition = AssertsOnly in
defm : BoolField<"misched-print-dags", "false",
  "Print schedule DAGs">;
defm : BoolField<"misched-regpressure", "true",
  "Enable register pressure scheduling">;
defm : ValueField<"misched-resource-cutoff", "unsigned", "10",
  "Number of intervals to track">;
let Condition = AssertsOnly in {
defm : BoolField<"misched-sort-resources-in-trace", "true",
  "Sort the resources printed in the dump trace">;
// In some situations a few uninteresting nodes depend on nearly all other
// nodes in the graph, provide a cutoff to hide them.
defm : ValueField<"misched-view-cutoff", "unsigned", "0",
  "Hide nodes with more predecessor/successor than cutoff">;
defm : BoolField<"misched-view-dags", "false",
  "Pop up a window to show MISched dags after they are processed">;
}
defm : BoolField<"verify-misched", "false",
  "Verify machine instrs before and after machine scheduling">;

// MachineSink.cpp
defm : ValueField<"machine-sink-cycle-limit", "unsigned", "50",
  "The maximum number of instructions considered for cycle sinking">;
defm : ValueField<"machine-sink-load-blocks-threshold", "unsigned", "20",
  "Do not try to find alias store for a load if the block number in the straight line is higher than this threshold">;
defm : ValueField<"machine-sink-load-instrs-threshold", "unsigned", "2000",
  "Do not try to find alias store for a load if there is a in-path block whose instruction number is higher than this threshold">;
defm : BoolField<"machine-sink-split", "true",
  "Split critical edges during machine sinking">;
defm : ValueField<"machine-sink-split-probability-threshold", "unsigned", "40",
  "Percentage threshold for splitting single-instruction critical edge. If the branch threshold is higher than this threshold, we allow speculative execution of up to 1 instruction to avoid branching to splitted critical edge">;
defm : BoolField<"sink-insts-to-avoid-spills", "false",
  "Sink instructions into cycles to avoid register spills">;

// MacroFusion.cpp
defm : BoolField<"misched-fusion", "true",
  "Enable scheduling for macro fusion">;

// ModuloSchedule.cpp
defm : BoolField<"pipeliner-swap-branch-targets-mve", "false",
  "Swap target blocks of a conditional branch for MVE expander">;

// PHIElimination.cpp
defm : BoolField<"phi-elim-edge-splitting", "true",
  "Enable critical edge splitting during PHI elimination">;
defm : BoolField<"phi-elim-live-out-early-exit", "true",
  "Use an early exit if isLiveOutPastPHIs returns true">;
defm : BoolField<"phi-elim-split-all-critical-edges", "false",
  "Split all critical edges during PHI elimination">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"no-phi-elim-live-out-early-exit", "phi-elim-live-out-early-exit=false">;

// PeepholeOptimizer.cpp
// Specify whether or not the value tracking looks through complex
// instructions. When this is false, the value tracker bails on everything that
// is not a copy or a bitcast.
defm : BoolField<"adv-copy-opt", "true",
  "Enable advanced copy optimization">;
defm : BoolField<"non-allocatable-phys-copy-opt", "true",
  "Enable non-allocatable physical register copy optimization">;
defm : BoolField<"peephole", "true",
  "Enable the peephole optimizer">;
defm : ValueField<"recurrence-chain-limit", "unsigned", "3",
  "Maximum length of recurrence chain when evaluating the benefit of commuting operands">;
defm : ValueField<"rewrite-phi-limit", "unsigned", "10",
  "Limit the length of PHI chains to lookup">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"disable-peephole", "peephole=false">;

// PostRASchedulerList.cpp
defm : OptionalValueField<"break-anti-dependencies", "StringRef",
  "Break post-RA scheduling anti-dependencies: \"critical\", \"all\", or \"none\"">;
// Post-RA scheduling is enabled with
// TargetSubtargetInfo.enablePostRAScheduler(). This flag can be used to
// override the target.
defm : OptionalBoolField<"post-RA-scheduler",
  "Enable scheduling after register allocation">;
let Condition = AssertsOnly in {
defm : ValueField<"postra-sched-debugdiv", "int", "0",
  "Debug control MBBs that are scheduled">;
defm : ValueField<"postra-sched-debugmod", "int", "0",
  "Debug control MBBs that are scheduled">;
}

// PreISelIntrinsicLowering.cpp
// Maximum bound of a memset with non-constant length for it to be expanded
// inline. Overrides TargetTransformInfo::getMaxBoundedMemSetInlineSize.
defm : OptionalValueField<"bounded-memset-inline-size", "uint64_t",
  "Set maximum bound of a variable-length memset to expand inline">;
// Threshold to leave statically sized memory intrinsic calls. Calls of known
// size larger than this will be expanded by the pass. Calls of unknown or
// lower size will be left for expansion in codegen.
defm : OptionalValueField<"mem-intrinsic-expand-size", "int64_t",
  "Set minimum mem intrinsic size to expand in IR">;

// RegisterCoalescer.cpp
// Temporary flag to test global copy optimization.
defm : OptionalBoolField<"join-globalcopies",
  "Coalesce copies that span blocks">;
defm : BoolField<"join-liveintervals", "true",
  "Coalesce copies">;
// Temporary flag to test critical edge unsplitting.
defm : BoolField<"join-splitedges", "false",
  "Coalesce copies on split edges">;
defm : ValueField<"large-interval-freq-threshold", "unsigned", "256",
  "For a large interval, if it is coalesced with other live intervals many times more than the threshold, stop its coalescing to control the compile time">;
defm : ValueField<"large-interval-size-threshold", "unsigned", "100",
  "If the valnos size of an interval is larger than the threshold, it is regarded as a large interval">;
defm : ValueField<"late-remat-update-threshold", "unsigned", "100",
  "During rematerialization for a copy, if the def instruction has many other copy uses to be rematerialized, delay the multiple separate live interval update work and do them all at once after all those rematerialization are done. It will save a lot of repeated work">;
defm : BoolField<"terminal-rule", "true",
  "Apply the terminal rule">;
defm : BoolField<"verify-coalescing", "false",
  "Verify machine instrs before and after register coalescing">;

// SafeStack.cpp
defm : BoolField<"safe-stack-coloring", "true",
  "enable safe stack coloring">;
// Use __safestack_pointer_address even if the platform has a faster way of
// access safe stack pointer.
defm : BoolField<"safe-stack-use-pointer-address", "false", "">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"safestack-use-pointer-address", "safe-stack-use-pointer-address">;

// SafeStackLayout.cpp
defm : BoolField<"safe-stack-layout", "true",
  "enable safe stack layout">;

// ScheduleDAG.cpp
let Condition = AssertsOnly in
defm : BoolField<"stress-sched", "false",
  "Stress test instruction scheduling">;

// ScheduleDAGInstrs.cpp
// Note: -dag-maps-huge-region and the -store-sequencing options might be used
// in tuning compile time vs output quality. Setting -dag-maps-huge-region so
// large that it will never be reached means best-effort, but may be slow.
// FIXME(@cofibrant): Drop in favour of store-sequencing.
// When the number of processed memops reaches this threshold, a reduction of
// maps will be done.
defm : ValueField<"dag-maps-huge-region", "unsigned", "500",
  "The limit to use while constructing the DAG prior to scheduling, at which point a trade-off is made to avoid excessive compile time. Ignored when store-sequencing is enabled">;
defm : OptionalBoolField<"sched-mi-aa",
  "Enable use of AA during MI DAG construction">;
defm : BoolField<"sched-mi-tbaa", "true",
  "Enable use of TBAA during MI DAG construction">;
let Condition = AssertsOnly in
defm : BoolField<"sched-print-cycles", "false",
  "Report top/bottom cycles when dumping SUnit instances">;
defm : BoolField<"scheditins", "true",
  "Use InstrItineraryData for latency lookup">;
defm : BoolField<"schedmodel", "true",
  "Use TargetSchedModel for latency lookup">;
defm : BoolField<"store-sequencing", "true",
  "Enable the store-sequencing DAG construction algorithm. This can eliminate a large number of redundant control dependencies and spurious alias analysis queries at the cost of some unnecessary dependencies">;
defm : ValueField<"store-sequencing-max-frontier-size", "unsigned", "500",
  "With store-sequencing DAG construction, after the number of memops in a frontier reaches this threshold, automatically promote the next store and clear the frontier">;
defm : ValueField<"store-sequencing-max-memops-in-flight", "unsigned", "3000",
  "With store-sequencing DAG construction, if the occupancy of all frontiers reaches this threshold, automatically promote the next memory operation to a global memory barrier and clear the frontiers">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"enable-aa-sched-mi", "sched-mi-aa">;
def : AliasOf<"enable-store-sequencing", "store-sequencing">;
def : AliasOf<"use-tbaa-in-sched-mi", "sched-mi-tbaa">;

// SelectOptimize.cpp
defm : ValueField<"cold-operand-max-cost-multiplier", "unsigned", "1",
  "Maximum cost multiplier of TCC_expensive for the dependence slice of a cold operand to be considered inexpensive">;
defm : ValueField<"cold-operand-threshold", "unsigned", "20",
  "Maximum frequency of path for an operand to be considered cold">;
defm : ValueField<"mispredict-default-rate", "unsigned", "25",
  "Default mispredict rate">;
defm : ValueField<"select-opti-loop-cycle-gain-threshold", "unsigned", "4",
  "Minimum gain per loop (in cycles) threshold">;
defm : ValueField<"select-opti-loop-gradient-gain-threshold", "unsigned", "25",
  "Gradient gain threshold (%)">;
defm : BoolField<"select-opti-loop-level-heuristics", "true",
  "Enable loop-level heuristics">;
defm : ValueField<"select-opti-loop-relative-gain-threshold", "unsigned", "8",
  "Minimum relative gain per loop threshold (1/X). Defaults to 12.5%">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"disable-loop-level-heuristics", "select-opti-loop-level-heuristics=false">;

// ShrinkWrap.cpp
defm : OptionalBoolField<"shrink-wrap",
  "Enable the shrink-wrapping pass">;
defm : BoolField<"shrink-wrap-region-split", "true",
  "Enable splitting of the restore block if possible">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"enable-shrink-wrap", "shrink-wrap">;

// StackColoring.cpp
defm : BoolField<"stack-coloring", "true",
  "Enable stack coloring">;
// Enable enhanced dataflow scheme for lifetime analysis (treat first use of
// stack slot as start of slot lifetime, as opposed to looking for
// LIFETIME_START marker). See "Implementation notes" in StackColoring.cpp for
// more info.
defm : BoolField<"stackcoloring-lifetime-start-on-first-use", "true",
  "Treat stack lifetimes as starting on first use, not on START marker">;
// The user may write code that uses allocas outside of the declared lifetime
// zone. This can happen when the user returns a reference to a local
// data-structure. We can detect these cases and decide not to optimize the
// code. If this flag is enabled, we try to save the user. This option is
// treated as overriding -stackcoloring-lifetime-start-on-first-use.
defm : BoolField<"stackcoloring-protect-from-escaped-allocas", "false",
  "Do not optimize lifetime zones that are broken">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"no-stack-coloring", "stack-coloring=false">;

// StackProtector.cpp
defm : BoolField<"stack-protector-check-noreturn-call", "true", "">;
defm : BoolField<"stack-protector-selectiondag", "true", "">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-check-noreturn-call", "stack-protector-check-noreturn-call=false">;
def : AliasOf<"enable-selectiondag-sp", "stack-protector-selectiondag">;

// StackSlotColoring.cpp
defm : ValueField<"ssc-dce-limit", "int", "-1", "">;
defm : BoolField<"stack-slot-sharing", "true",
  "Enable slot sharing during stack slot coloring">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"no-stack-slot-sharing", "stack-slot-sharing=false">;

// TailDuplicator.cpp
defm : ValueField<"tail-dup-indirect-size", "unsigned", "20",
  "Maximum instructions to consider tail duplicating blocks that end with indirect branches">;
defm : ValueField<"tail-dup-limit", "unsigned", "~0U", "">;
defm : ValueField<"tail-dup-pred-size", "unsigned", "16",
  "Maximum predecessors (maximum successors at the same time) to consider tail duplicating blocks">;
// Heuristic for tail duplication.
defm : ValueField<"tail-dup-size", "unsigned", "2",
  "Maximum instructions to consider tail duplicating">;
defm : ValueField<"tail-dup-succ-size", "unsigned", "16",
  "Maximum successors (maximum predecessors at the same time) to consider tail duplicating blocks">;
defm : BoolField<"tail-dup-verify", "false",
  "Verify sanity of PHI instructions during taildup">;

// TargetPassConfig.cpp
defm : BoolField<"basic-block-section-match-infer", "false",
  "Enable matching and inference when generating basic block sections">;
defm : BoolField<"codegen-pipeline-trigger-crash", "false",
  "Trigger crash in codegen pipeline">;
defm : OptionalBoolField<"debugify-and-strip-all-safe",
  "Debugify MIR before and Strip debug after each pass except those known to be unsafe when debug info is present">;
defm : OptionalBoolField<"debugify-check-and-strip-all-safe",
  "Debugify MIR before, by checking and stripping the debug info after, each pass except those known to be unsafe when debug info is present">;
defm : BoolField<"disable-atexit-based-global-dtor-lowering", "false",
  "For MachO, disable atexit()-based global destructor lowering">;
defm : BoolField<"disable-block-placement", "false",
  "Disable probability-driven block placement">;
defm : BoolField<"disable-branch-fold", "false",
  "Disable branch folding">;
// Disable the pass to fix unwind information. Whether the pass is included in
// the pipeline is controlled via the target options, this option serves as
// manual override.
defm : BoolField<"disable-cfi-fixup", "false",
  "Disable the CFI fixup pass">;
defm : BoolField<"disable-cgp", "false",
  "Disable Codegen Prepare">;
defm : BoolField<"disable-constant-hoisting", "false",
  "Disable ConstantHoisting">;
defm : BoolField<"disable-copyprop", "false",
  "Disable Copy Propagation pass">;
defm : BoolField<"disable-early-ifcvt", "false",
  "Disable Early If-conversion">;
defm : BoolField<"disable-early-taildup", "false",
  "Disable pre-register allocation tail duplication">;
defm : BoolField<"disable-expand-reductions", "false",
  "Disable the expand reduction intrinsics pass from running">;
// Disable MIRProfileLoader before BlockPlacement. This is for debugging and
// tuning purposes.
defm : BoolField<"disable-layout-fsprofile-loader", "false",
  "Disable MIRProfileLoader before BlockPlacement">;
defm : BoolField<"disable-lsr", "false",
  "Disable Loop Strength Reduction Pass">;
defm : BoolField<"disable-machine-cse", "false",
  "Disable Machine Common Subexpression Elimination">;
defm : BoolField<"disable-machine-dce", "false",
  "Disable Machine Dead Code Elimination">;
defm : BoolField<"disable-machine-licm", "false",
  "Disable Machine LICM">;
defm : BoolField<"disable-machine-sink", "false",
  "Disable Machine Sinking">;
defm : BoolField<"disable-mir-output-verify", "false",
  "Do not verify machine code at the end of a pipeline that stops before code emission">;
defm : BoolField<"disable-partial-libcall-inlining", "false",
  "Disable Partial Libcall Inlining">;
defm : BoolField<"disable-post-ra", "false",
  "Disable Post Regalloc Scheduler">;
defm : BoolField<"disable-postra-machine-licm", "false",
  "Disable Machine LICM">;
defm : BoolField<"disable-postra-machine-sink", "false",
  "Disable PostRA Machine Sinking">;
// Disable MIRProfileLoader before RegAlloc. This is for debugging and tuning
// purpose.
defm : BoolField<"disable-ra-fsprofile-loader", "false",
  "Disable MIRProfileLoader before RegAlloc">;
defm : BoolField<"disable-replace-with-vec-lib", "false",
  "Disable replace with vector math call pass">;
defm : BoolField<"disable-select-optimize", "true",
  "Disable the select-optimization pass from running">;
defm : BoolField<"disable-ssc", "false",
  "Disable Stack Slot Coloring">;
defm : BoolField<"disable-tail-duplicate", "false",
  "Disable tail duplication">;
defm : BoolField<"enable-block-placement-stats", "false",
  "Collect probability-driven block placement stats">;
defm : BoolField<"enable-gc-empty-basic-blocks", "false",
  "Enable garbage-collecting empty basic blocks">;
defm : BoolField<"enable-global-merge-func", "false",
  "Enable global merge functions that are based on hash function">;
defm : BoolField<"enable-implicit-null-checks", "false",
  "Fold null checks into faulting memory operations">;
defm : OptionalBoolField<"enable-ipra",
  "Enable interprocedural register allocation to reduce load/store at procedure calls">;
defm : OptionalBoolField<"enable-pipeliner",
  "Enable the MachinePipeliner (overrides the target's default)">;
defm : EnumField<"function-splitting", "std::optional<FunctionSplittingMode>", "std::nullopt",
  "none,bbsections,all",
  ["FunctionSplittingMode::None", "FunctionSplittingMode::BBSectionsOnly",
   "FunctionSplittingMode::All"],
  "Which functions are eligible for late function splitting">;
// -fast-isel selects FastISel at any optimization level, and -fast-isel=0
// disables it at -O0, where it is used unless GlobalISel is enabled.
defm : OptionalBoolField<"fast-isel",
  "Enable the \"fast\" instruction selector">;
let MetaVarName = "<filename>" in
defm : ValueField<"fs-profile-file", "StringRef", "",
  "Flow Sensitive profile file name">;
let MetaVarName = "<filename>" in
defm : ValueField<"fs-remapping-file", "StringRef", "",
  "Flow Sensitive profile remapping file name">;
defm : OptionalBoolField<"global-isel",
  "Enable the \"global\" instruction selector">;
defm : EnumField<"global-isel-abort", "std::optional<GlobalISelAbortMode>", "std::nullopt",
  "0,1,2", ["GlobalISelAbortMode::Disable", "GlobalISelAbortMode::Enable",
            "GlobalISelAbortMode::DisableWithDiag"],
  "Enable abort calls when \"global\" instruction selection fails to lower/select an instruction">;
// Temporary option to allow experimenting with MachineScheduler as a post-RA
// scheduler. Targets can "properly" enable this with
// substitutePass(&PostRASchedulerID, &PostMachineSchedulerID).
// Targets can return true in targetSchedulesPostRAScheduling() and
// insert a PostRA scheduling pass wherever it wants.
defm : BoolField<"misched-postra", "false",
  "Run MachineScheduler post regalloc (independent of preRA sched)">;
// FIXME: remove this after switching to NPM or GlobalISel, whichever gets there
// first...
defm : BoolField<"print-after-isel", "false",
  "Print machine instrs after ISel">;
defm : BoolField<"print-isel-input", "false",
  "Print LLVM IR input to isel pass">;
defm : BoolField<"print-regusage", "false",
  "Print register usage details collected for IPRA">;
defm : OptionalBoolField<"regalloc-fast-tied",
  "Have the fast register allocator lower tied operands itself instead of running TwoAddressInstructionPass (overrides the target's default)">;
defm : BoolField<"split-static-data", "false",
  "Split static data sections into hot and cold sections using profile information">;
let MetaVarName = "<pass-name>" in
defm : ValueField<"start-after", "StringRef", "",
  "Resume compilation after a specific pass">;
let MetaVarName = "<pass-name>" in
defm : ValueField<"start-before", "StringRef", "",
  "Resume compilation before a specific pass">;
let MetaVarName = "<pass-name>" in
defm : ValueField<"stop-after", "StringRef", "",
  "Stop compilation after a specific pass">;
let MetaVarName = "<pass-name>" in
defm : ValueField<"stop-before", "StringRef", "",
  "Stop compilation before a specific pass">;
defm : OptionalBoolField<"verify-machineinstrs",
  "Verify generated machine code">;

// TwoAddressInstructionPass.cpp
defm : BoolField<"twoaddr-analyze-revcopy-tied", "true",
  "Analyze tied operands when looking for reversed copy chain">;
defm : ValueField<"twoaddr-dataflow-edge-limit", "unsigned", "10",
  "Maximum number of dataflow edges to traverse when evaluating the benefit of commuting operands">;
// Temporary flag to disable rescheduling.
defm : BoolField<"twoaddr-reschedule", "true",
  "Coalesce copies by rescheduling">;

// VLIWMachineScheduler.cpp
// Check if the scheduler should penalize instructions that are available to
// early due to a zero-latency dependence.
defm : BoolField<"vliw-misched-check-early-avail", "true", "">;
defm : BoolField<"vliw-misched-ignore-bb-reg-pressure", "false", "">;
// This value is used to determine if a register class is a high pressure set.
// We compute the maximum number of registers needed and divided by the total
// available. Then, we compare the result to this value.
defm : ValueField<"vliw-misched-reg-pressure", "float", "0.75f",
  "High register pressure threhold">;
defm : BoolField<"vliw-misched-use-newer-candidate", "true", "">;
defm : ValueField<"vliw-misched-verbose-level", "unsigned", "1", "">;

// WinEHPrepare.cpp
defm : BoolField<"win-eh-prepare-cleanups", "true",
  "Remove implausible terminators and do other similar cleanups">;
defm : BoolField<"win-eh-prepare-demotion", "true",
  "Demote cross scopes after cloning multicolor basic blocks">;

// WindowScheduler.cpp
defm : ValueField<"window-diff-limit", "unsigned", "2",
  "The lower limit of the difference between best II and base II in the window algorithm. If the difference is smaller than this lower limit, window scheduling will not be performed">;
defm : ValueField<"window-ii-coeff", "unsigned", "5",
  "The coefficient used when initializing II in the window algorithm">;
defm : ValueField<"window-ii-limit", "unsigned", "1000",
  "The upper limit of II in the window algorithm">;
defm : ValueField<"window-region-limit", "unsigned", "3",
  "The lower limit of the scheduling region in the window algorithm">;
defm : ValueField<"window-search-num", "unsigned", "6",
  "The number of searches per loop in the window algorithm. 0 means no search number limit">;
defm : ValueField<"window-search-ratio", "unsigned", "40",
  "The ratio of searches per loop in the window algorithm. 100 means search all positions in the loop, while 0 means not performing any search">;
