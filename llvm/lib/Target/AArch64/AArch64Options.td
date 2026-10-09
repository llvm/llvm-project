//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def AArch64Options : OptionsStruct<prefix = "aarch64-">;

// AArch64A57FPLoadBalancing.cpp
// Enforce the algorithm to use the scavenged register even when the original
// destination register is the correct color. Used for testing.
defm : BoolField<"aarch64-a57-fp-load-balancing-force-all", "false",
  "Always modify dest registers regardless of color">;
// Never use the balance information obtained from chains - return a specific
// color always. Used for testing.
defm : ValueField<"aarch64-a57-fp-load-balancing-override", "unsigned", "0",
  "Ignore balance information, always return (1: Even, 2: Odd)">;

// AArch64AdvSIMDScalarPass.cpp
// Allow forcing all i64 operations with equivalent SIMD instructions to use
// them. For stress-testing the transformation function.
defm : BoolField<"aarch64-simd-scalar-force-all", "false",
  "Force use of AdvSIMD scalar instructions everywhere">;

// AArch64Arm64ECCallLowering.cpp
defm : BoolField<"arm64ec-generate-thunks", "true",
  "">;
defm : BoolField<"arm64ec-lower-direct-to-indirect", "true",
  "">;

// AArch64AsmPrinter.cpp
defm : EnumField<"aarch64-ptrauth-auth-checks", "std::optional<AArch64PAuth::PtrauthCheckMode>", "std::nullopt",
  "none,poison,trap", ["AArch64PAuth::PtrauthCheckMode::Unchecked", "AArch64PAuth::PtrauthCheckMode::Poison", "AArch64PAuth::PtrauthCheckMode::Trap"],
  "Check pointer authentication auth/resign failures">;

// AArch64CodeLayoutOpt.cpp
defm : ValueField<"aarch64-code-layout-opt-align-functions", "unsigned", "64",
  "Function alignment in bytes for code layout optimization (must be a power of 2)">;

// AArch64FrameLowering.cpp
defm : BoolField<"aarch64-disable-multivector-spill-fill", "false",
  "Disable use of LD/ST pairs for SME2 or SVE2p1">;
defm : BoolField<"aarch64-order-frame-objects", "true",
  "sort stack allocations">;
defm : BoolField<"aarch64-redzone", "false",
  "enable use of redzone on AArch64">;
defm : BoolField<"aarch64-split-sve-objects", "true",
  "Split allocation of ZPR & PPR objects">;
// Whether to insert padding into non-streaming functions (for testing).
defm : BoolField<"aarch64-stack-hazard-in-non-streaming", "false",
  "">;
// Stack hazard size for analysis remarks. StackHazardSize takes precedence.
defm : ValueField<"aarch64-stack-hazard-remark-size", "unsigned", "0",
  "">;
defm : BoolField<"homogeneous-prolog-epilog", "false",
  "Emit homogeneous prologue and epilogue for the size optimization (default = off)">;
defm : BoolField<"stack-tagging-merge-settag", "true",
  "merge settag instruction in function epilog">;

// AArch64InstrInfo.cpp
defm : ValueField<"aarch64-b-offset-bits", "unsigned", "26",
  "Restrict range of B instructions (DEBUG)">;
defm : ValueField<"aarch64-bcc-offset-bits", "unsigned", "19",
  "Restrict range of Bcc instructions (DEBUG)">;
defm : ValueField<"aarch64-cb-offset-bits", "unsigned", "9",
  "Restrict range of CB instructions (DEBUG)">;
defm : ValueField<"aarch64-cbz-offset-bits", "unsigned", "19",
  "Restrict range of CB[N]Z instructions (DEBUG)">;
defm : BoolField<"aarch64-outliner-compact-unwind-frame", "true",
  "Use a frame record for Mach-O non-leaf outlined functions">;
defm : ValueField<"aarch64-search-limit", "unsigned", "2048",
  "Restrict range of instructions to search for the machine-combiner gather pattern optimization">;
defm : ValueField<"aarch64-tbz-offset-bits", "unsigned", "14",
  "Restrict range of TB[N]Z instructions (DEBUG)">;

// AArch64ISelLowering.cpp
defm : ValueField<"aarch64-br-merging-base-cost", "int", "2",
  "Cost threshold for merging multiple conditionals into one branch versus splitting into multiple branches: conditionals are merged when their instruction cost is below this limit and split above it. Set to -1 to never merge branches">;
defm : ValueField<"aarch64-br-merging-cbz-tbnz-bias", "int", "6",
  "Decreases 'aarch64-br-merging-base-cost' when a condition can lower to a single CBZ/CBNZ or TBZ/TBNZ compare-and-branch, to bias toward splitting. Set to 0 to disable">;
defm : ValueField<"aarch64-br-merging-ccmp-bias", "int", "6",
  "Increases 'aarch64-br-merging-base-cost' to account for the CCMP instruction, which is always available on AArch64 and makes merging branch conditions cheaper">;
defm : ValueField<"aarch64-br-merging-likely-bias", "int", "0",
  "Increases 'aarch64-br-merging-base-cost' when all conditionals are likely to be executed, biasing toward merging. Set to -1 to never merge likely branches">;
defm : ValueField<"aarch64-br-merging-unlikely-bias", "int", "-1",
  "Decreases 'aarch64-br-merging-base-cost' when all conditionals are unlikely to be executed, biasing toward splitting. Set to -1 to never merge unlikely branches">;
// FIXME: The necessary dtprel relocations don't seem to be supported
// well in the GNU bfd and gold linkers at the moment. Therefore, by
// default, for now, fall back to GeneralDynamic code generation.
defm : BoolField<"aarch64-elf-ldtls-generation", "false",
  "Allow AArch64 Local Dynamic TLS code generation">;
defm : BoolField<"aarch64-enable-ext-to-tbl", "true",
  "Combine ext and trunc to TBL">;
// By turning this on, we will not fallback to DAG ISel when encountering
// scalable vector types for all instruction, even if SVE is not yet supported
// with some instructions.
// See [AArch64TargetLowering::fallbackToDAGISel] for implementation details.
defm : BoolField<"aarch64-enable-gisel-sve", "false",
  "Enable / disable SVE scalable vectors in Global ISel">;
defm : BoolField<"aarch64-enable-logical-imm", "true",
  "Enable AArch64 logical imm instruction optimization">;
// Temporary option added for the purpose of testing functionality added
// to DAGCombiner.cpp in D92230. It is expected that this can be removed
// in future when both implementations will be based off MGATHER rather
// than the GLD1 nodes added for the SVE gather load intrinsics.
defm : BoolField<"aarch64-enable-mgather-combine", "true",
  "Combine extends of AArch64 masked gather intrinsics">;
// All of the XOR, OR and CMP use ALU ports, and data dependency will become the
// bottleneck after this transform on high end CPU. So this max leaf node
// limitation is guard cmp+ccmp will be profitable.
defm : ValueField<"aarch64-max-xors", "unsigned", "16",
  "Maximum of xors">;
// Development flag to allow incremental bring up. Will be removed once the
// implementation is complete.
defm : BoolField<"aarch64-sve-vls-bfloat-support", "false",
  "Use SVE for fixed-length vector bfloat operations">;
// FPMR writes might be a synchronization barrier and thus carry a significant
// cost. Give users the option to skip writes when the requested value is
// already set.
defm : BoolField<"aarch64-use-conditional-fpmr-write", "false",
  "Only write FPMR when the requested value differs from the current value">;
// TODO: This option should be removed once we switch to always using PTRADD in
// the SelectionDAG.
defm : BoolField<"aarch64-use-featcpa-codegen", "false",
  "Generate ISD::PTRADD nodes for pointer arithmetic in SelectionDAG for FEAT_CPA">;

// AArch64LoadStoreOptimizer.cpp
// Limits how far we search for const offset instructions
// when we form index address load/store instructions.
defm : ValueField<"aarch64-load-store-const-scan-limit", "unsigned", "10",
  "">;
// Enable register renaming to find additional store pairing opportunities.
defm : BoolField<"aarch64-load-store-renaming", "true",
  "">;
// Limits how far we search for load/store pairs.
defm : ValueField<"aarch64-load-store-scan-limit", "unsigned", "20",
  "">;
// Limits how far back we scan from a GPR store to find a
// UMOV that can be folded into a direct FPR store.
defm : ValueField<"aarch64-umov-fold-scan-limit", "unsigned", "16",
  "">;
// Limits how far we search for update instructions when we form
// pre-/post-index instructions.
defm : ValueField<"aarch64-update-scan-limit", "unsigned", "100",
  "">;

// AArch64LowerHomogeneousPrologEpilog.cpp
defm : ValueField<"frame-helper-size-threshold", "int", "2",
  "The minimum number of instructions that are outlined in a frame helper (default = 2)">;

// AArch64PointerAuth.cpp
defm : EnumField<"aarch64-cfi-llvm-set-ra-sign-state", "AArch64PAuth::SetRAStateMode", "AArch64PAuth::SetRAStateMode::PAuthLR",
  "never,pauth-lr,always", ["AArch64PAuth::SetRAStateMode::Never", "AArch64PAuth::SetRAStateMode::PAuthLR", "AArch64PAuth::SetRAStateMode::Always"],
  "Control emission of .cfi_set_ra_state for PAC return address signing CFI">;

// AArch64PromoteConstant.cpp
// Stress testing mode - disable heuristics.
defm : BoolField<"aarch64-stress-promote-const", "false",
  "Promote all vector constants">;

// AArch64PTrueCoalescing.cpp
defm : BoolField<"aarch64-enable-ptrue-coalescing", "true",
  "Enable coalescing of compatible AArch64 SVE PTRUE instructions">;

// AArch64SelectionDAGInfo.cpp
defm : BoolField<"aarch64-lower-to-sme-routines", "true",
  "Enable AArch64 SME memory operations to lower to librt functions">;
defm : BoolField<"aarch64-use-mops", "true",
  "Enable AArch64 MOPS instructions for memcpy/memset/memmove">;

// AArch64SpeculationHardening.cpp
defm : BoolField<"aarch64-slh-loads", "true",
  "Sanitize loads from memory">;

// AArch64StackTagging.cpp
defm : OptionalBoolField<"stack-tagging-merge-init",
  "merge stack variable initializers with tagging when possible">;
defm : ValueField<"stack-tagging-merge-init-scan-limit", "unsigned", "40",
  "">;
defm : ValueField<"stack-tagging-merge-init-size-limit", "unsigned", "272",
  "">;
defm : EnumField<"stack-tagging-record-stack-history", "std::optional<AArch64::StackTaggingRecordStackHistoryMode>", "std::nullopt",
  "none,instr", ["AArch64::StackTaggingRecordStackHistoryMode::None", "AArch64::StackTaggingRecordStackHistoryMode::Instr"],
  "Record stack frames with tagged allocations in a thread-local ring buffer">;
defm : OptionalBoolField<"stack-tagging-use-stack-safety",
  "Use Stack Safety analysis results">;

// AArch64StackTaggingPreRA.cpp
defm : BoolField<"stack-tagging-first-slot-opt", "true",
  "Apply first slot optimization for stack tagging (eliminate ADDG Rt, Rn, 0, 0)">;
defm : EnumField<"stack-tagging-unchecked-ld-st", "AArch64::UncheckedLdStMode", "AArch64::UncheckedLdStMode::Safe",
  "never,safe,always", ["AArch64::UncheckedLdStMode::Never", "AArch64::UncheckedLdStMode::Safe", "AArch64::UncheckedLdStMode::Always"],
  "Unconditionally apply unchecked-ld-st optimization (even for large stack frames, or in the presence of variable sized allocas)">;

// AArch64Subtarget.cpp
defm : EnumField<"aarch64-authenticated-lr-check-method", "std::optional<AArch64PAuth::AuthCheckMethod>", "std::nullopt",
  "none,load,high-bits-notbi,xpac,xpac-hint", ["AArch64PAuth::AuthCheckMethod::None", "AArch64PAuth::AuthCheckMethod::DummyLoad", "AArch64PAuth::AuthCheckMethod::HighBitsNoTBI", "AArch64PAuth::AuthCheckMethod::XPAC", "AArch64PAuth::AuthCheckMethod::XPACHint"],
  "Override the variant of check applied to authenticated LR during tail call">;
defm : BoolField<"aarch64-early-ifcvt", "true",
  "Enable the early if converter pass">;
// Subreg liveness tracking is disabled by default for now until all issues
// are ironed out. This option allows the feature to be used in tests.
defm : BoolField<"aarch64-enable-subreg-liveness-tracking", "false",
  "Enable subreg liveness tracking">;
defm : ValueField<"aarch64-insert-extract-base-cost", "std::optional<unsigned>", "std::nullopt",
  "Base cost of vector insert/extract element">;
defm : BoolField<"aarch64-macho-enable-nonlazybind", "false",
  "Call nonlazybind functions via direct GOT load for Mach-O">;
defm : ValueField<"aarch64-min-jump-table-entries", "std::optional<unsigned>", "std::nullopt",
  "Set minimum number of entries to use a jump table on AArch64">;
defm streaming_hazard_size : ValueField<"aarch64-streaming-hazard-size",
  "std::optional<unsigned>", "std::nullopt",
  "Hazard size for streaming mode memory accesses. 0 = disabled">;
def : SeparateOrEq<["-", "--"], "aarch64-stack-hazard-size">,
  Alias<streaming_hazard_size>;
defm : BoolField<"aarch64-use-aa", "true",
  "Enable the use of AA during codegen">;
// If OS supports TBI, use this flag to enable it.
defm : BoolField<"aarch64-use-tbi", "false",
  "Assume that top byte of an address is ignored">;
defm : OptionalBoolField<"sve-use-scalar-inc-vl",
  "Prefer add+cnt over addvl/inc/dec">;
defm : ValueField<"sve-vscale-for-tuning", "std::optional<unsigned>", "std::nullopt",
  "Force a vscale for tuning factor for SVE">;

// AArch64TargetMachine.cpp
defm : BoolField<"aarch64-enable-atomic-cfg-tidy", "true",
  "Run SimplifyCFG after expanding atomic operations to make use of cmpxchg flow-based information">;
defm : BoolField<"aarch64-enable-branch-relax", "true",
  "Relax out of range conditional branches">;
defm : BoolField<"aarch64-enable-branch-targets", "true",
  "Enable the AArch64 branch target pass">;
defm : BoolField<"aarch64-enable-ccmp", "true",
  "Enable the CCMP formation pass">;
defm : BoolField<"aarch64-enable-collect-loh", "true",
  "Enable the pass that emits the linker optimization hints (LOH)">;
defm : BoolField<"aarch64-enable-compress-jump-tables", "true",
  "Use smallest entry possible for jump tables">;
defm : BoolField<"aarch64-enable-cond-br-tune", "true",
  "Enable the conditional branch tuning pass">;
defm : BoolField<"aarch64-enable-condopt", "true",
  "Enable the condition optimizer pass">;
defm : BoolField<"aarch64-enable-copy-propagation", "true",
  "Enable the copy propagation with AArch64 copy instr">;
defm : BoolField<"aarch64-enable-copyelim", "true",
  "Enable the redundant copy elimination pass">;
defm : BoolField<"aarch64-enable-dead-defs", "true",
  "Enable the pass that removes dead definitions and replaces stores to them with stores to the zero register">;
defm : BoolField<"aarch64-enable-early-ifcvt", "true",
  "Run early if-conversion">;
defm : BoolField<"aarch64-enable-falkor-hwpf-fix", "true",
  "">;
defm : BoolField<"aarch64-enable-gep-opt", "false",
  "Enable optimizations on complex GEPs">;
defm : BoolField<"aarch64-enable-gisel-ldst-postlegal", "false",
  "Enable GlobalISel's post-legalizer load/store optimization pass">;
defm : BoolField<"aarch64-enable-gisel-ldst-prelegal", "true",
  "Enable GlobalISel's pre-legalizer load/store optimization pass">;
defm : ValueField<"aarch64-enable-global-isel-at-O", "int", "0",
  "Enable GlobalISel at or below an opt level (-1 to disable)">;
// FIXME: Unify control over GlobalMerge.
defm : OptionalBoolField<"aarch64-enable-global-merge",
  "Enable the global merge pass">;
defm : BoolField<"aarch64-enable-ldst-opt", "true",
  "Enable the load/store pair optimization pass">;
defm : BoolField<"aarch64-enable-loop-data-prefetch", "true",
  "Enable the loop data prefetch pass">;
defm : BoolField<"aarch64-enable-mcr", "true",
  "Enable the machine combiner pass">;
defm : BoolField<"aarch64-enable-pipeliner", "false",
  "Enable Machine Pipeliner for AArch64">;
defm : BoolField<"aarch64-enable-predicate-as-counter-loop-rewrites", "false",
  "Enable rewriting loops with wide loop-carried masks to use predicate-as-counter">;
defm : BoolField<"aarch64-enable-promote-const", "true",
  "Enable the promote constant pass">;
defm : BoolField<"aarch64-enable-simd-scalar", "false",
  "Enable use of AdvSIMD scalar integer instructions">;
defm : BoolField<"aarch64-enable-sink-fold", "true",
  "Enable sinking and folding of instruction copies">;
defm : BoolField<"aarch64-enable-stp-suppress", "true",
  "Suppress STP for AArch64">;
defm : BoolField<"aarch64-enable-sve-shuffle-opts", "true",
  "Enable pattern matching of shuffles that could make use of SVE instructions like tbl or the bottom/top variants">;
defm : BoolField<"aarch64-select-opt", "true",
  "Enable select to branch optimizations">;
defm : BoolField<"aarch64-srlt-mitigate-sr2r", "true",
  "Enable SUBREG_TO_REG mitigation by adding 'implicit-def' for super-regs when using Subreg Liveness Tracking">;
defm : ValueField<"aarch64-sve-vector-bits-max", "unsigned", "0",
  "Assume SVE vector registers are at most this big, with zero meaning no maximum size is assumed">;
defm : ValueField<"aarch64-sve-vector-bits-min", "unsigned", "0",
  "Assume SVE vector registers are at least this big, with zero meaning no minimum size is assumed">;
defm : BoolField<"enable-aarch64-sme-peephole-opt", "true",
  "Perform SME peephole optimization">;
defm : BoolField<"force-streaming", "false",
  "Force the use of streaming code for all functions">;
defm : BoolField<"force-streaming-compatible", "false",
  "Force the use of streaming-compatible code for all functions">;

// AArch64TargetObjectFile.cpp
defm : BoolField<"aarch64-emit-debug-tls-location", "false",
  "Emit the TLS DWARF location with DTPREL relocation for AArch64">;

// AArch64TargetTransformInfo.cpp
// A complete guess as to a reasonable cost.
defm : ValueField<"aarch64-base-histcnt-cost", "unsigned", "8",
  "The cost of a histcnt instruction">;
defm : ValueField<"aarch64-force-unroll-threshold", "int", "0",
  "Threshold for forced unrolling of small loops in AArch64">;
defm : ValueField<"call-penalty-sm-change", "unsigned", "5",
  "Penalty of calling a function that requires a change to PSTATE.SM">;
defm : ValueField<"dmb-lookahead-threshold", "unsigned", "10",
  "The number of instructions to search for a redundant dmb">;
defm : BoolField<"enable-aarch64-lsr-cost-opt", "true",
  "">;
defm : BoolField<"enable-aarch64-or-like-select", "true",
  "">;
defm : BoolField<"enable-falkor-hwpf-unroll-fix", "true",
  "">;
// Experimental option that will only be fully functional when the
// code-generator is changed to use SVE instead of NEON for all fixed-width
// operations.
defm : BoolField<"enable-fixedwidth-autovec-in-streaming-mode", "false",
  "">;
// Experimental option that will only be fully functional when the cost-model
// and code-generator have been changed to avoid using scalable vector
// instructions that are not legal in streaming SVE mode.
defm : BoolField<"enable-scalable-autovec-in-streaming-mode", "false",
  "">;
defm : ValueField<"inline-call-penalty-sm-change", "unsigned", "10",
  "Penalty of inlining a call that requires a change to PSTATE.SM">;
defm : ValueField<"neon-nonconst-stride-overhead", "unsigned", "10",
  "">;
defm : ValueField<"sve-gather-overhead", "std::optional<unsigned>", "std::nullopt",
  "">;
defm : OptionalBoolField<"sve-prefer-fixed-over-scalable-if-equal",
  "">;
defm : ValueField<"sve-scatter-overhead", "std::optional<unsigned>", "std::nullopt",
  "">;
defm : ValueField<"sve-tail-folding-insn-threshold", "unsigned", "15",
  "">;

// GISel/AArch64PostLegalizerCombiner.cpp
defm : BoolField<"aarch64-postlegalizer-consecutive-memops", "true",
  "Enable consecutive memop optimization in AArch64PostLegalizerCombiner">;
