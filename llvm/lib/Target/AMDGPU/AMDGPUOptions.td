//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def AMDGPUOptions : OptionsStruct<prefix = "amdgpu-">;

// AMDGPUAlwaysInlinePass.cpp
defm : BoolField<"amdgpu-stress-function-calls", "false",
  "Force all functions to be noinline">;

// AMDGPUAttributor.cpp
defm : ValueField<"amdgpu-indirect-call-specialization-threshold", "unsigned", "3",
  "A threshold controls whether an indirect call will be specialized">;

// AMDGPUBarrierLatency.cpp
defm : ValueField<"amdgpu-barrier-signal-wait-latency", "unsigned", "16",
  "Synthetic latency between S_BARRIER_SIGNAL and S_BARRIER_WAIT to encourage scheduling independent work between them">;

// AMDGPUCoExecSchedStrategy.cpp
defm : OptionalEnumField<"amdgpu-block-carried-latency", "AMDGPU::CarriedLatency", "off,fence,all",
  ["AMDGPU::CarriedLatency::Off", "AMDGPU::CarriedLatency::Fence", "AMDGPU::CarriedLatency::All"],
  "Estimate block-carried latency and include it in the effective candidate stall cost">;

// AMDGPUCodeGenPrepare.cpp
defm : BoolField<"amdgpu-codegenprepare-break-large-phis", "true",
  "Break large PHI nodes for DAGISel">;
defm : ValueField<"amdgpu-codegenprepare-break-large-phis-threshold", "unsigned", "32",
  "Minimum type size in bits for breaking large PHI nodes">;
// Legalize 64-bit division by using the generic IR expansion.
defm : BoolField<"amdgpu-codegenprepare-expand-div64", "false",
  "Expand 64-bit division in AMDGPUCodeGenPrepare">;
// Disabled to better test the backend implementations of fdiv.
defm : BoolField<"amdgpu-codegenprepare-fdiv-expansion", "true",
  "Expand floating point division in AMDGPUCodeGenPrepare">;
defm : BoolField<"amdgpu-codegenprepare-force-break-large-phis", "false",
  "For testing purposes, always break large PHIs even if it isn't profitable">;
// Disabled to leave all integer division operations as they are. This
// supersedes amdgpu-codegenprepare-expand-div64 and is used for testing the
// legalizer.
defm : BoolField<"amdgpu-codegenprepare-idiv-expansion", "true",
  "Expand integer division in AMDGPUCodeGenPrepare">;
defm : BoolField<"amdgpu-codegenprepare-mul24", "true",
  "Introduce mul24 intrinsics in AMDGPUCodeGenPrepare">;
defm : BoolField<"amdgpu-codegenprepare-widen-constant-loads", "false",
  "Widen sub-dword constant address space loads in AMDGPUCodeGenPrepare">;

// AMDGPUHSAMetadataStreamer.cpp
defm : BoolField<"amdgpu-dump-hsa-metadata", "false",
  "Dump AMDGPU HSA Metadata">;
defm : BoolField<"amdgpu-verify-hsa-metadata", "false",
  "Verify AMDGPU HSA Metadata">;

// AMDGPUIGroupLP.cpp
defm : BoolField<"amdgpu-igrouplp-exact-solver", "false",
  "Whether to use the exponential time solver to fit the instructions to the pipeline as closely as possible">;
defm : BoolField<"amdgpu-igrouplp-exact-solver-cost-heur", "true",
  "Whether to use the cost heuristic to make choices as we traverse the search space using the exact solver. Defaulted to on, and if turned off, we will use the node order -- attempting to put the later nodes in the later sched groups. Experimentally, results are mixed, so this should be set on a case-by-case basis">;
defm : ValueField<"amdgpu-igrouplp-exact-solver-cutoff", "unsigned", "0",
  "The maximum number of scheduling group conflicts which we attempt to solve with the exponential time exact solver. Problem sizes greater than this will be solved by the less accurate greedy algorithm. Selecting solver by size is superseded by manually selecting the solver (e.g. by amdgpu-igrouplp-exact-solver)">;
defm : ValueField<"amdgpu-igrouplp-exact-solver-max-branches", "uint64_t", "0",
  "The amount of branches that we are willing to explore with the exact algorithm before giving up">;

// AMDGPUISelLowering.cpp
defm : BoolField<"amdgpu-bypass-slow-div", "true",
  "Skip 64-bit divide for dynamic 32-bit values">;

// AMDGPULateCodeGenPrepare.cpp
// Scalar load widening needs running after load-store-vectorizer as that pass
// doesn't handle overlapping cases. In addition, this pass enhances the
// widening to handle cases where scalar sub-dword loads are naturally aligned
// only but not dword aligned.
defm : BoolField<"amdgpu-late-codegenprepare-widen-constant-loads", "true",
  "Widen sub-dword constant address space loads in AMDGPULateCodeGenPrepare">;

// AMDGPULegalizerInfo.cpp
// Hack until load/store selection patterns support any tuple of legal types.
defm : BoolField<"amdgpu-global-isel-new-legality", "false",
  "Use GlobalISel desired legality, rather than try to use rules compatible with selection patterns">;

// AMDGPULibCalls.cpp
defm : BoolField<"amdgpu-prelink", "false",
  "Enable pre-link mode optimizations">;

// AMDGPULibFunc.cpp
defm : BoolField<"amdgpu-enable-ocl-mangling-mismatch-workaround", "true",
  "Enable the workaround for OCL name mangling mismatch">;

// AMDGPULowerModuleLDSPass.cpp
defm : EnumField<"amdgpu-lower-module-lds-strategy", "AMDGPU::LDSLoweringKind", "AMDGPU::LDSLoweringKind::hybrid", "table,module,kernel,hybrid",
  ["AMDGPU::LDSLoweringKind::table", "AMDGPU::LDSLoweringKind::module", "AMDGPU::LDSLoweringKind::kernel", "AMDGPU::LDSLoweringKind::hybrid"],
  "Specify lowering strategy for function LDS access">;
defm : BoolField<"amdgpu-super-align-lds-globals", "true",
  "Increase alignment of LDS if it is not on align boundary">;

// AMDGPUNextUseAnalysis.cpp
// 'graphics' and 'compute' modes arose due to initial competing implementations
// of next-use analysis that emphasized different types of workloads. This
// implementation is a compromise that combines aspects of both. Over time, the
// hope is we will be able to remove some of these differences and settle on a
// more unified implementation.
defm : ValueField<"amdgpu-next-use-analysis-config", "StringRef", "\"graphics\"",
  "Config preset: 'graphics' or 'compute'">;
defm : OptionalBoolField<"amdgpu-next-use-analysis-count-phis",
  "Count PHI instructions toward distance and block size">;
defm : BoolField<"amdgpu-next-use-analysis-distance-cache", "true",
  "Enable live-reg-use distance cache">;
defm : OptionalValueField<"amdgpu-next-use-analysis-dump-distance-as-json", "StringRef", "">;
defm : BoolField<"amdgpu-next-use-analysis-dump-distance-def-to-use", "false", "">;
defm : BoolField<"amdgpu-next-use-analysis-dump-distance-verbose", "false", "">;
defm : OptionalBoolField<"amdgpu-next-use-analysis-forward-only",
  "Restrict inter-block distances to forward-reachable paths">;
defm : OptionalBoolField<"amdgpu-next-use-analysis-precise-use-modeling",
  "Model PHI uses via incoming edge block with loop-aware reachability filtering">;
defm : OptionalBoolField<"amdgpu-next-use-analysis-use-preheader-model",
  "Promote loop-entry and inner-loop uses to the loop preheader">;

// AMDGPUPerfHintAnalysis.cpp
defm : ValueField<"amdgpu-indirect-access-weight", "unsigned", "1000",
  "Indirect access memory instruction weight">;
defm : ValueField<"amdgpu-large-stride-threshold", "unsigned", "64",
  "Large stride memory access threshold">;
defm : ValueField<"amdgpu-large-stride-weight", "unsigned", "1000",
  "Large stride memory access weight">;
defm : ValueField<"amdgpu-limit-wave-threshold", "unsigned", "50",
  "Kernel limit wave threshold in %">;
defm : ValueField<"amdgpu-membound-threshold", "unsigned", "50",
  "Function mem bound threshold in %">;

// AMDGPUPreloadKernelArguments.cpp
defm : BoolField<"amdgpu-kernarg-preload", "true",
  "Enable preload kernel arguments to SGPRs">;
defm : ValueField<"amdgpu-kernarg-preload-count", "unsigned", "0",
  "How many kernel arguments to preload onto SGPRs">;

// AMDGPUPromoteAlloca.cpp
defm : BoolField<"amdgpu-promote-alloca-to-lds", "true",
  "Promote alloca to LDS">;
defm : BoolField<"amdgpu-promote-alloca-to-vector", "true",
  "Promote alloca to vector">;
defm : ValueField<"amdgpu-promote-alloca-to-vector-limit", "unsigned", "0",
  "Maximum byte size to consider promote alloca to vector">;
defm : OptionalValueField<"amdgpu-promote-alloca-to-vector-max-regs", "unsigned",
  "Maximum vector size (in 32b registers) to use when promoting alloca">;
// Use up to 1/4 of available register budget for vectorization. FIXME: Increase
// the limit for whole function budgets? Perhaps x2?
defm : OptionalValueField<"amdgpu-promote-alloca-to-vector-vgpr-ratio", "unsigned",
  "Ratio of VGPRs to budget for promoting alloca to vectors">;
defm : ValueField<"amdgpu-promote-alloca-vector-loop-user-weight", "unsigned", "4",
  "The bonus weight of users of allocas within loop when sorting profitable allocas">;

// AMDGPUResourceUsageAnalysis.cpp
defm : OptionalValueField<"amdgpu-assume-dynamic-stack-object-size", "uint32_t",
  "Assumed extra stack use if there are any variable sized objects (in bytes)">;
// In code object v4 and older, we need to tell the runtime some amount ahead of
// time if we don't know the true stack size. Assume a smaller number if this is
// only due to dynamic / non-entry block allocas.
defm : OptionalValueField<"amdgpu-assume-external-call-stack-size", "uint32_t",
  "Assumed stack use of any external call (in bytes)">;

// AMDGPURewriteOutArguments.cpp
defm : BoolField<"amdgpu-any-address-space-out-arguments", "false",
  "Replace pointer out arguments with struct returns for non-private address space">;
defm : ValueField<"amdgpu-max-return-arg-num-regs", "unsigned", "16",
  "Approximately limit number of return registers for replacing out arguments">;

// AMDGPUSetWavePriority.cpp
defm : ValueField<"amdgpu-set-wave-priority-valu-insts-threshold", "unsigned", "100",
  "VALU instruction count threshold for adjusting wave priority">;

// AMDGPUSplitModule.cpp
let Condition = AssertsOnly in
defm : BoolField<"amdgpu-module-splitting-debug-proposal-search", "false",
  "print all proposals received and whether they were rejected or accepted">;
defm : BoolField<"amdgpu-module-splitting-externalize-address-taken", "true",
  "externalize functions whose addresses are taken">;
defm : BoolField<"amdgpu-module-splitting-externalize-globals", "true",
  "externalize global variables with local linkage; disabling may cause globals to be duplicated which increases binary size">;
defm : ValueField<"amdgpu-module-splitting-large-threshold", "float", "2.0f",
  "when max depth is reached and we can no longer branch out, this value determines if a function is worth merging into an already existing partition to reduce code duplication. This is a factor of the ideal partition size, e.g. 2.0 means we consider the function for merging if its cost (including its callees) is 2x the size of an ideal partition">;
defm : ValueField<"amdgpu-module-splitting-max-depth", "unsigned", "8",
  "maximum search depth. 0 forces a greedy approach. warning: the algorithm is up to O(2^N), where N is the max depth">;
defm : ValueField<"amdgpu-module-splitting-merge-threshold", "float", "0.7f",
  "when a function is considered for merging into a partition that already contains some of its callees, do the merge if at least n% of the code it can reach is already present inside the partition; e.g. 0.7 means only merge >70%">;
defm : ValueField<"amdgpu-module-splitting-print-module-dotcfg", "StringRef", "",
  "output file to write out the dotgraph representation of the input module">;
defm : ValueField<"amdgpu-module-splitting-print-partition-summaries", "StringRef", "",
  "output file to write out a summary of the partitions created for each module">;
let Condition = AssertsOnly in
defm : BoolField<"amdgpu-module-splitting-serial-execution", "false",
  "use a lock file so only one process in the system can run this pass at once. useful to avoid mangled debug output in multithreaded environments">;

// AMDGPUSwLowerLDS.cpp
defm : BoolField<"amdgpu-asan-instrument-lds", "true",
  "Run asan instrumentation on LDS instructions lowered to global memory">;

// AMDGPUTargetMachine.cpp
defm : EnumField<"amdgpu-atomic-optimizer-strategy", "AMDGPU::ScanOptions", "AMDGPU::ScanOptions::Iterative", "DPP,Iterative,None",
  ["AMDGPU::ScanOptions::DPP", "AMDGPU::ScanOptions::Iterative", "AMDGPU::ScanOptions::None"],
  "Select DPP or Iterative strategy for scan">;
defm : BoolField<"amdgpu-attributor-enable", "true",
  "Enable AMDGPUAttributorPass">;
// Option is used in lit tests to prevent deadcoding of patterns inspected.
defm : BoolField<"amdgpu-dce-in-ra", "true",
  "Enable machine DCE inside regalloc">;
defm : BoolField<"amdgpu-dpp-combine", "true",
  "Enable DPP combiner">;
defm : BoolField<"amdgpu-early-ifcvt", "false",
  "Run early if-conversion">;
// Option to inline all early.
defm : BoolField<"amdgpu-early-inline-all", "false",
  "Inline all functions early">;
// Enable address space based alias analysis.
defm : BoolField<"amdgpu-enable-aa", "true",
  "Enable AMDGPU Alias Analysis">;
// Enable GFX11+ s_delay_alu insertion.
defm : OptionalBoolField<"amdgpu-enable-delay-alu",
  "Enable s_delay_alu insertion">;
defm : BoolField<"amdgpu-enable-hipstdpar", "false",
  "Enable HIP Standard Parallelism Offload support">;
defm : OptionalBoolField<"amdgpu-enable-image-intrinsic-optimizer",
  "Enable image intrinsic optimizer pass">;
defm : BoolField<"amdgpu-enable-lower-exec-sync", "true",
  "Enable lowering of execution synchronization">;
defm : BoolField<"amdgpu-enable-pipeliner", "false",
  "Enable Machine Pipeliner for AMDGCN">;
defm : OptionalBoolField<"amdgpu-enable-pre-ra-optimizations",
  "Enable Pre-RA optimizations pass">;
defm : BoolField<"amdgpu-enable-promote-kernel-arguments", "true",
  "Enable promotion of flat kernel pointer arguments to global">;
defm : BoolField<"amdgpu-enable-remove-incompatible-functions", "true",
  "Enable removal of functions when they use features not supported by the target GPU">;
defm : BoolField<"amdgpu-enable-rewrite-partial-reg-uses", "true",
  "Enable rewrite partial reg uses pass">;
defm : BoolField<"amdgpu-enable-sw-lower-lds", "true",
  "Enable lowering of lds to global memory pass and asan instrument resulting IR">;
defm : BoolField<"amdgpu-enable-uniform-intrinsic-combine", "true",
  "Enable/Disable the Uniform Intrinsic Combine Pass">;
// Enable GFX11+ VOPD.
defm : OptionalBoolField<"amdgpu-enable-vopd",
  "Enable VOPD, dual issue of VALU in wave32">;
// Option to run internalize pass.
defm : BoolField<"amdgpu-internalize-symbols", "false",
  "Enable elimination of non-kernel functions and unused globals">;
defm : BoolField<"amdgpu-ir-lower-kernel-arguments", "true",
  "Lower kernel argument loads in IR pass">;
defm : BoolField<"amdgpu-link-time-closed-world", "false",
  "Whether has closed-world assumption at link time">;
// Option to disable vectorizer for tests.
defm : OptionalBoolField<"amdgpu-load-store-vectorizer",
  "Enable load store vectorizer">;
defm : BoolField<"amdgpu-loop-prefetch", "false",
  "Enable loop data prefetch on AMDGPU">;
defm : BoolField<"amdgpu-lower-global-ctor-dtor", "true",
  "Lower GPU ctor / dtors to globals on the device">;
defm : BoolField<"amdgpu-opt-exec-mask-pre-ra", "true",
  "Run pre-RA exec mask optimizations">;
defm : BoolField<"amdgpu-opt-vgpr-liverange", "true",
  "Enable VGPR liverange optimizations for if-else structure">;
defm : BoolField<"amdgpu-reassign-regs", "true",
  "Enable register reassign optimizations on gfx10+">;
defm : OptionalBoolField<"amdgpu-scalar-ir-passes",
  "Enable scalar IR passes">;
defm : ValueField<"amdgpu-sched-strategy", "StringRef", "",
  "Select custom AMDGPU scheduling strategy">;
defm : OptionalBoolField<"amdgpu-sdwa-peephole",
  "Enable SDWA peepholer">;
defm : BoolField<"amdgpu-set-wave-priority", "false",
  "Adjust wave priority">;
// Enable lib calls simplifications.
defm : BoolField<"amdgpu-simplify-libcall", "true",
  "Enable amdgpu library simplifications">;
defm : OptionalBoolField<"amdgpu-sramecc",
  "Force amdgpu.sramecc for testing">;
defm : BoolField<"amdgpu-use-ssa-machine-scheduler", "false",
  "Use the machine scheduler in SSA mode">;
defm : OptionalBoolField<"amdgpu-xnack",
  "Force amdgpu.xnack value for testing">;

// AMDGPUTargetTransformInfo.cpp
defm : ValueField<"amdgpu-inline-arg-alloca-cost", "unsigned", "4000",
  "Cost of alloca argument">;
// If the amount of scratch memory to eliminate exceeds our ability to allocate
// it into registers we gain nothing by aggressively inlining functions for that
// heuristic.
defm : ValueField<"amdgpu-inline-arg-alloca-cutoff", "unsigned", "256",
  "Maximum alloca size to use for inline cost">;
// Inliner constraint to achieve reasonable compilation time.
defm : ValueField<"amdgpu-inline-max-bb", "size_t", "1100",
  "Maximum number of BBs allowed in a function after inlining (compile time constraint)">;
// This default unroll factor is based on microbenchmarks on gfx1030.
defm : ValueField<"amdgpu-memcpy-loop-unroll", "unsigned", "16",
  "Unroll factor (affecting 4x32-bit operations) to use for memory operations when lowering statically-sized memcpy, memmove, or memset as a loop">;
defm : ValueField<"amdgpu-unroll-max-block-to-analyze", "unsigned", "32",
  "Inner loop block size threshold to analyze in unroll for AMDGPU">;
defm : BoolField<"amdgpu-unroll-runtime-local", "true",
  "Allow runtime unroll for AMDGPU if local memory used in a loop">;
defm : ValueField<"amdgpu-unroll-threshold-if", "unsigned", "200",
  "Unroll threshold increment for AMDGPU for each if statement inside loop">;
defm : ValueField<"amdgpu-unroll-threshold-local", "unsigned", "1000",
  "Unroll threshold for AMDGPU if local memory used in a loop">;
defm : ValueField<"amdgpu-unroll-threshold-private", "unsigned", "2700",
  "Unroll threshold for AMDGPU if private memory used in a loop">;

// AMDGPUWaitSGPRHazards.cpp
defm : OptionalBoolField<"amdgpu-sgpr-hazard-boundary-cull",
  "Cull hazards on function boundaries">;
defm : OptionalBoolField<"amdgpu-sgpr-hazard-mem-wait-cull",
  "Cull hazards on memory waits">;
defm : OptionalValueField<"amdgpu-sgpr-hazard-mem-wait-cull-threshold", "unsigned",
  "Number of tracked SGPRs before initiating hazard cull on memory wait">;

// AMDGPUWaitcntUtils.cpp
defm : OptionalBoolField<"amdgpu-expert-scheduling-mode",
  "Enable expert scheduling mode 2 for all functions (GFX12+ only)">;

// GCNHazardRecognizer.cpp
// This is intended for debugging purposes only.
defm : ValueField<"amdgpu-snop-padding", "unsigned", "0",
  "Insert a s_nop x before every instruction">;
defm : BoolField<"amdgpu-wmma-vnop-hoisting", "true",
  "Hoist WMMA hazard V_NOPs from loops to preheaders">;

// GCNPreRAAntiHints.cpp
defm : ValueField<"amdgpu-addr-anti-hint-window", "unsigned", "16",
  "Number of later memory instructions an address anti-hint stays open">;
defm : BoolField<"amdgpu-anti-hints-for-addr", "true",
  "Enable Anti-Hints for memory address operands and subsequent VGPR writes to avoid wait xcnt">;
defm : BoolField<"amdgpu-anti-hints-for-va-vdst", "false",
  "Enable Anti-Hints for VA-VDST">;
defm : EnumListField<"amdgpu-anti-hints-rules", "AMDGPU::AntiHintRule", "none,mfma-waw,mfma-war,wmma-war-ab,swmmac-war-index,wmma-waw,trans-war,mem-addr-war,va-vdst-war,all",
  ["AMDGPU::AntiHintRule::None", "AMDGPU::AntiHintRule::MFMAWAW", "AMDGPU::AntiHintRule::MFMAWAR", "AMDGPU::AntiHintRule::WMMAWARAB", "AMDGPU::AntiHintRule::SWMMACWARIndex", "AMDGPU::AntiHintRule::WMMAWAW", "AMDGPU::AntiHintRule::TransWAR", "AMDGPU::AntiHintRule::MemAddrWAR", "AMDGPU::AntiHintRule::VAVdstWAR", "AMDGPU::AntiHintRule::All"],
  "Anti-hints rules to select">;
defm : ValueField<"amdgpu-va-vdst-lookback-window", "unsigned", "32",
  "Lookback window for VA_VDST anti-hints">;

// GCNPreRALongBranchReg.cpp
defm : ValueField<"amdgpu-long-branch-factor", "double", "1.0",
  "Factor to apply to what qualifies as a long branch to reserve a pair of scalar registers. If this value is 0 the long branch registers are never reserved. As this value grows the greater chance the branch distance will fall within the threshold and the registers will be marked to be reserved. We lean towards always reserving a register for long jumps">;

// GCNPreRAOptimizations.cpp
defm : BoolField<"amdgpu-anti-hints", "true",
  "Enable register allocation anti-hints">;

// GCNRegPressure.cpp
defm : BoolField<"amdgpu-print-rp-downward", "false",
  "Use GCNDownwardRPTracker for GCNRegPressurePrinter pass">;

// GCNSchedStrategy.cpp
defm : BoolField<"amdgpu-clustered-low-occupancy-reschedule", "true",
  "Enable clustered low occupancy rescheduling for ILP scheduling stage">;
defm : ValueField<"amdgpu-lirp-instant-lower-bound", "unsigned", "10",
  "Lower bound (percent of the VGPR excess limit) on instant RP, below which a region is skipped">;
defm : BoolField<"amdgpu-lirp-reschedule", "true",
  "Enable live interval RP reschedule stage">;
defm : ValueField<"amdgpu-lirp-threshold", "unsigned", "10",
  "Percent increase of live interval RP over instant pressure to trigger rescheduling">;
defm : ValueField<"amdgpu-lirp-vgpr-reduction", "unsigned", "90",
  "Reduction factor (percent) for VGPR threshold during live interval RP reschedule stage">;
let Condition = AssertsOnly in {
defm : BoolField<"amdgpu-print-max-reg-pressure-regusage-after-scheduler", "false",
  "Print a list of live registers along with their def/uses at the point of maximum register pressure after scheduling">;
defm : BoolField<"amdgpu-print-max-reg-pressure-regusage-before-scheduler", "false",
  "Print a list of live registers along with their def/uses at the point of maximum register pressure before scheduling">;
}
defm : BoolField<"amdgpu-rewrite-mfma-form-sched-stage", "false",
  "Enable mfma rewrite scheduling stage">;
defm : ValueField<"amdgpu-schedule-metric-bias", "unsigned", "10",
  "Sets the bias which adds weight to occupancy vs latency. Set it to 100 to chase the occupancy only">;
defm : BoolField<"amdgpu-schedule-relaxed-occupancy", "false",
  "Relax occupancy targets for kernels which are memory bound (amdgpu-membound-threshold), or Wave Limited (amdgpu-limit-wave-threshold)">;
defm : ValueField<"amdgpu-scheduler-pending-queue-limit", "unsigned", "256",
  "Max (Available+Pending) size to inspect pending queue (0 disables)">;
defm : BoolField<"amdgpu-unclustered-high-rp-reschedule", "true",
  "Enable unclustered high register pressure reduction scheduling stage">;
defm : OptionalBoolField<"amdgpu-use-amdgpu-trackers",
  "Use the AMDGPU specific RPTrackers during scheduling">;

// GCNSubtarget.cpp
defm : OptionalValueField<"amdgpu-nsa-threshold", "unsigned",
  "Number of addresses from which to enable MIMG NSA">;
defm : BoolField<"amdgpu-use-aa-in-codegen", "true",
  "Enable the use of AA during codegen">;
defm : BoolField<"amdgpu-vgpr-index-mode", "false",
  "Use GPR indexing mode instead of movrel for vector indexing">;

// R600TargetMachine.cpp
defm : BoolField<"r600-if-convert", "true",
  "Use if conversion pass">;
defm : BoolField<"r600-ir-structurize", "true",
  "Use StructurizeCFG IR pass">;

// SIFixSGPRCopies.cpp
defm : BoolField<"amdgpu-enable-merge-m0", "true",
  "Merge and hoist M0 initializations">;

// SIFormMemoryClauses.cpp
// Clauses longer then 15 instructions would overflow one of the counters and
// stall. They can stall even earlier if there are outstanding counters.
defm : ValueField<"amdgpu-max-memory-clause", "unsigned", "15",
  "Maximum length of a memory clause, instructions">;

// SIFrameLowering.cpp
defm : BoolField<"amdgpu-spill-vgpr-to-agpr", "true",
  "Enable spilling VGPRs to AGPRs">;

// SIISelLowering.cpp
defm : BoolField<"amdgpu-loop-alignment", "true",
  "Align and prefetch loops">;
defm : BoolField<"amdgpu-use-divergent-register-indexing", "false",
  "Use indirect register addressing for divergent indexes">;

// SIInsertHardClauses.cpp
defm : OptionalValueField<"amdgpu-hard-clause-length-limit", "unsigned",
  "Maximum number of memory instructions to place in the same hard clause">;

// SIInsertWaitcnts.cpp
defm : BoolField<"amdgpu-waitcnt-forcezero", "false",
  "Force all waitcnt instrs to be emitted as s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)">;
defm : BoolField<"amdgpu-waitcnt-load-forcezero", "false",
  "Force all waitcnt load counters to wait until 0">;

// SIInstrInfo.cpp
defm : BoolField<"amdgpu-fix-16-bit-physreg-copies", "true",
  "Fix copies between 32 and 16 bit registers by extending to 32 bit">;
// Must be at least 4 to be able to branch over minimum unconditional branch
// code. This is only for making it possible to write reasonably small tests for
// long branches.
defm : ValueField<"amdgpu-s-branch-bits", "unsigned", "16",
  "Restrict range of branch instructions (DEBUG)">;

// SILowerControlFlow.cpp
defm : BoolField<"amdgpu-remove-redundant-endcf", "true", "">;

// SILowerSGPRSpills.cpp
defm : ValueField<"amdgpu-num-vgprs-for-wwm-alloc", "unsigned", "10",
  "Max num VGPRs for whole-wave register allocation">;

// SIMemoryLegalizer.cpp
defm : BoolField<"amdgpu-skip-cache-invalidations", "false",
  "Use this to skip inserting cache invalidating instructions">;

// SIPreAllocateWWMRegs.cpp
defm : BoolField<"amdgpu-prealloc-sgpr-spill-vgprs", "false", "">;

// SIRegisterInfo.cpp
defm : BoolField<"amdgpu-spill-cfi-saved-regs", "false",
  "Enable spilling the registers required for CFI emission">;
defm : BoolField<"amdgpu-spill-sgpr-to-vgpr", "true",
  "Enable spilling SGPRs to VGPRs">;
defm : OptionalValueField<"amdgpu-stress-agpr", "unsigned",
  "Limit AGPRs to N registers by reserving the rest">;
defm : OptionalValueField<"amdgpu-stress-sgpr", "unsigned",
  "Limit SGPRs to N registers by reserving the rest">;
defm : OptionalValueField<"amdgpu-stress-vgpr", "unsigned",
  "Limit VGPRs to N registers by reserving the rest">;

// TODO: Remove the old spellings after 2027-04.
def : AliasOf<"amdgcn-skip-cache-invalidations", "amdgpu-skip-cache-invalidations">;
def : AliasOf<"amdgpu-codegenprepare-disable-fdiv-expansion", "amdgpu-codegenprepare-fdiv-expansion=false">;
def : AliasOf<"amdgpu-codegenprepare-disable-idiv-expansion", "amdgpu-codegenprepare-idiv-expansion=false">;
def : AliasOf<"amdgpu-disable-clustered-low-occupancy-reschedule", "amdgpu-clustered-low-occupancy-reschedule=false">;
def : AliasOf<"amdgpu-disable-loop-alignment", "amdgpu-loop-alignment=false">;
def : AliasOf<"amdgpu-disable-rewrite-mfma-form-sched-stage", "amdgpu-rewrite-mfma-form-sched-stage=false">;
def : AliasOf<"amdgpu-disable-unclustered-high-rp-reschedule", "amdgpu-unclustered-high-rp-reschedule=false">;
def : AliasOf<"amdgpu-module-splitting-no-externalize-address-taken", "amdgpu-module-splitting-externalize-address-taken=false">;
def : AliasOf<"amdgpu-module-splitting-no-externalize-globals", "amdgpu-module-splitting-externalize-globals=false">;
def : AliasOf<"disable-promote-alloca-to-lds", "amdgpu-promote-alloca-to-lds=false">;
def : AliasOf<"disable-promote-alloca-to-vector", "amdgpu-promote-alloca-to-vector=false">;
def : AliasOf<"enable-amdgpu-aa", "amdgpu-enable-aa">;
def : AliasOf<"promote-alloca-vector-loop-user-weight", "amdgpu-promote-alloca-vector-loop-user-weight">;
