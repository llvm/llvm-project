//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def CodeGenOptions : OptionsStruct;

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
def : AliasOf<"disable-ifcvt-forked-diamond", "ifcvt-forked-diamond=false">;
def : AliasOf<"disable-ifcvt-simple", "ifcvt-simple=false">;
def : AliasOf<"disable-ifcvt-simple-false", "ifcvt-simple-false=false">;
def : AliasOf<"disable-ifcvt-triangle", "ifcvt-triangle=false">;
def : AliasOf<"disable-ifcvt-triangle-false", "ifcvt-triangle-false=false">;
def : AliasOf<"disable-ifcvt-triangle-rev", "ifcvt-triangle-rev=false">;

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
// Number of times to re-run the outliner. This is not the total number of runs
// as the outliner will run at least one time. The default value is set to 0,
// meaning the outliner will run one time and rerun zero times after that.
defm : ValueField<"machine-outliner-reruns", "unsigned", "0",
  "Number of times to rerun the outliner after the initial outline">;
defm : ValueField<"outliner-benefit-threshold", "unsigned", "1",
  "The minimum size in bytes before an outlining candidate is accepted">;
defm : BoolField<"outliner-leaf-descendants", "true",
  "Consider all leaf descendants of internal nodes of the suffix tree as candidates for outlining (if false, only leaf children are considered)">;
// TODO: Remove the old spelling after 2027-05.
def : AliasOf<"disable-global-outlining", "global-outlining=false">;

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
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-adv-copy-opt", "adv-copy-opt=false">;
def : AliasOf<"disable-non-allocatable-phys-copy-opt", "non-allocatable-phys-copy-opt=false">;
def : AliasOf<"disable-peephole", "peephole=false">;

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
