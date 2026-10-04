//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def CodeGenOptions : OptionsStruct;

// CodeGenPrepare.cpp
defm : BoolField<"bbsections-guided-section-prefix", "true",
  "Use the basic-block-sections profile to determine the text section prefix for hot functions. "
  "Functions with basic-block-sections profile will be placed in `.text.hot` regardless of their FDO profile info. "
  "Other functions won't be impacted, i.e., their prefixes will be decided by FDO/sampleFDO profiles">;
defm : BoolField<"addr-sink-combine-base-gv", "true",
  "Allow combining of BaseGV field in Address sinking">;
defm : BoolField<"addr-sink-combine-base-offs", "true",
  "Allow combining of BaseOffs field in Address sinking">;
defm : BoolField<"addr-sink-combine-base-reg", "true",
  "Allow combining of BaseReg field in Address sinking">;
defm : BoolField<"addr-sink-combine-scaled-reg", "true",
  "Allow combining of ScaledReg field in Address sinking">;
defm : BoolField<"addr-sink-new-phis", "false",
  "Allow creation of Phis in Address sinking">;
defm : BoolField<"addr-sink-new-select", "true",
  "Allow creation of selects in Address sinking">;
defm : OptionalBoolField<"addr-sink-using-gep",
  "Address sinking in CGP using GEPs">;
defm : BoolField<"enable-andcmp-sinking", "true",
  "Enable sinking and/cmp into branches">;
defm : BoolField<"disable-cgp-branch-opts", "false",
  "Disable branch optimizations in CodeGenPrepare">;
defm : BoolField<"disable-complex-addr-modes", "false",
  "Disables combining addressing modes with different parts in optimizeMemoryInst">;
defm : BoolField<"disable-cgp-delete-phis", "false",
  "Disable elimination of dead PHI nodes">;
defm : BoolField<"disable-cgp-ext-ld-promotion", "false",
  "Disable ext(promotable(ld)) -> promoted(ext(ld)) optimization in CodeGenPrepare">;
defm : BoolField<"force-split-store", "false",
  "Force store splitting no matter what the target query says">;
defm : ValueField<"cgp-freq-ratio-to-skip-merge", "uint64_t", "2",
  "Skip merging empty blocks if (frequency of empty block) / (frequency of destination block) is greater than this ratio">;
defm : BoolField<"disable-cgp-gc-opts", "false",
  "Disable GC optimizations in CodeGenPrepare">;
defm : ValueField<"cgpp-huge-func", "unsigned", "10000",
  "Least BB number of huge function">;
defm : BoolField<"cgp-icmp-eq2icmp-st", "false",
  "Enable ICMP_EQ to ICMP_S(L|G)T conversion">;
defm : ValueField<"cgp-max-address-users-to-scan", "unsigned", "100",
  "Max number of address users to look at">;
defm : BoolField<"cgp-optimize-phi-types", "true",
  "Enable converting phi types in CodeGenPrepare">;
defm : BoolField<"disable-preheader-prot", "false",
  "Disable protection against removing loop preheaders">;
defm : BoolField<"disable-cgp-select2branch", "false",
  "Disable select to branch conversion">;
defm : BoolField<"cgp-split-large-offset-gep", "true",
  "Enable splitting large offset of GEP">;
defm : BoolField<"disable-cgp-store-extract", "false",
  "Disable store(extract) optimizations in CodeGenPrepare">;
defm : BoolField<"stress-cgp-ext-ld-promotion", "false",
  "Stress test ext(promotable(ld)) -> promoted(ext(ld)) optimization in CodeGenPrepare">;
defm : BoolField<"stress-cgp-store-extract", "false",
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
defm : BoolField<"enable-split-machine-functions", "false",
  "Split out cold blocks from machine functions based on profile information">;
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
