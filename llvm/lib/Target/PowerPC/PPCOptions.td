//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def PPCOptions : OptionsStruct<prefix = "ppc-">;

// PPCAsmPrinter.cpp
defm : BoolField<"ppc-aix-ssp-tb-bit", "false",
  "Enable Passing SSP Canary info in Trackback on AIX">;
defm : BoolField<"ppc-ifunc-local-if-proven", "false",
  "During ifunc lowering, the compiler assumes the resolver returns dso-local functions and bails out if non-local functions are detected; this flag flips the assumption: resolver returns preemptible functions unless the compiler can prove all paths return local functions">;
// this flag is used for testing only as it might generate bad code.
defm : BoolField<"ppc-test-ifunc-warn-noerror", "false", "">;

// PPCFrameLowering.cpp
defm : BoolField<"ppc-enable-pe-vector-spills", "false",
  "Enable spills in prologue to vector registers">;

// PPCInstrInfo.cpp
defm : BoolField<"ppc-cmp-opt", "true",
  "Enable compare instruction optimization">;
defm : BoolField<"ppc-crash-on-vsx-self-copy", "false",
  "Causes the backend to crash instead of generating a nop VSX copy">;
defm : BoolField<"ppc-ctr-loop-analysis", "true",
  "Enable analysis for CTR loops">;
defm : ValueField<"ppc-fma-rp-factor", "float", "1.5",
  "register pressure factor for the transformations">;
defm : BoolField<"ppc-fma-rp-reduction", "true",
  "enable register pressure reduce in machine combiner pass">;
defm : BoolField<"ppc-old-latency-calc", "false",
  "Use the old (incorrect) instruction latency calculation">;

// PPCISelDAGToDAG.cpp
defm : BoolField<"ppc-bit-perm-rewriter-stress-rotates", "false",
  "stress rotate selection in aggressive ppc isel for bit permutations">;
// FIXME: Remove this once the bug has been fixed!
defm : BoolField<"ppc-expose-andi-glue-bug", "false",
  "expose the ANDI glue bug on PPC">;
defm : OptionalEnumField<"ppc-gpr-icmps", "PPC::ICmpInGPRType",
  "none,all,i32,i64,nonextin,zext,zexti32,zexti64,sext,sexti32,sexti64",
  ["PPC::ICmpInGPRType::None", "PPC::ICmpInGPRType::All",
   "PPC::ICmpInGPRType::I32", "PPC::ICmpInGPRType::I64",
   "PPC::ICmpInGPRType::NonExtIn", "PPC::ICmpInGPRType::Zext",
   "PPC::ICmpInGPRType::ZextI32", "PPC::ICmpInGPRType::ZextI64",
   "PPC::ICmpInGPRType::Sext", "PPC::ICmpInGPRType::SextI32",
   "PPC::ICmpInGPRType::SextI64"],
  "Specify the types of comparisons to emit GPR-only code for">;
defm : BoolField<"ppc-tls-opt", "true",
  "Enable tls optimization peephole">;
defm : BoolField<"ppc-use-bit-perm-rewriter", "true",
  "use aggressive ppc isel for bit permutations">;
defm : BoolField<"ppc-use-branch-hint", "true",
  "Enable static hinting of branches on ppc">;

// PPCISelLowering.cpp
defm : ValueField<"ppc-aix-shared-lib-tls-model-opt-limit", "unsigned", "1",
  "Set inclusive limit count of TLS local-dynamic access(es) in a function to use initial-exec">;
defm : OptionalBoolField<"ppc-auto-paired-vec-st",
  "enable automatically generated 32byte paired vector stores">;
defm : ValueField<"ppc-gather-alias-max-depth", "unsigned", "18",
  "max depth when checking alias info in GatherAllAliases()">;
defm : BoolField<"ppc-ilp-pref", "true",
  "Set the node scheduling preference to ILP on PPC">;
defm : BoolField<"ppc-innermost-loop-align32", "true",
  "Align innermost loops to 32 bytes on ppc">;
defm : ValueField<"ppc-min-bit-test-cmps", "unsigned", "3",
  "Set minimum of largest number of comparisons to use bit test for switch on PPC">;
defm : ValueField<"ppc-min-jump-table-entries", "unsigned", "64",
  "Set minimum number of entries to use a jump table on PPC">;
defm : BoolField<"ppc-p10-store-forward", "true",
  "enable P10 store forward-friendly conversion">;
defm : BoolField<"ppc-perfect-shuffle", "false",
  "enable vector permute decomposition">;
defm : BoolField<"ppc-preinc", "true",
  "enable preincrement load/store generation on PPC">;
defm : BoolField<"ppc-sco", "true",
  "enable sibling call optimization on ppc">;
defm : BoolField<"ppc-unaligned", "true",
  "enable unaligned load/store generation on PPC">;
defm : BoolField<"ppc-use-absolute-jumptables", "false",
  "use absolute jump tables on ppc">;

// PPCLoopInstrFormPrep.cpp
// Commoning chain will reduce the register pressure, so we don't consider about
// the PHI nodes number. But commoning chain will increase the addi/add number
// in the loop and also increase loop ILP. Maximum chain number should be same
// with hardware IssueWidth, because we won't benefit from ILP if the parallel
// chains number is bigger than IssueWidth. We assume there are 2 chains in one
// bucket, so there would be 4 buckets at most on P9(IssueWidth is 8).
defm : ValueField<"ppc-chaincommon-max-vars", "unsigned", "4",
  "Bucket number per loop for PPC loop chain common">;
defm : ValueField<"ppc-chaincommon-min-threshold", "unsigned", "4",
  "Minimal common base load/store instructions triggering chain commoning preparation. Must be not smaller than 4">;
// If would not be profitable if the common base has only one load/store, ISEL
// should already be able to choose best load/store form based on offset for
// single load/store. Set minimal profitable value default to 2 and make it as
// an option.
defm : ValueField<"ppc-dispprep-min-threshold", "unsigned", "2",
  "Minimal common base load/store instructions triggering DS/DQ form preparation">;
// The sum of the ppc-dqprep-max-vars, ppc-dsprep-max-vars, and
// ppc-preinc-prep-max-vars thresholds for all loops can not be larger than
// ppc-formprep-max-vars. The thresholds are experimental values on Power9.
defm : ValueField<"ppc-dqprep-max-vars", "unsigned", "8",
  "Potential PHI threshold per loop for PPC loop prep of DQ form">;
defm : ValueField<"ppc-dsprep-max-vars", "unsigned", "3",
  "Potential PHI threshold per loop for PPC loop prep of DS form">;
defm : BoolField<"ppc-formprep-chain-commoning", "false",
  "Enable chain commoning in PPC loop prepare pass">;
defm : ValueField<"ppc-formprep-max-vars", "unsigned", "24",
  "Potential common base number threshold per function for PPC loop prep">;
defm : BoolField<"ppc-formprep-prefer-update", "true",
  "prefer update form when ds form is also a update form">;
defm : BoolField<"ppc-formprep-update-nonconst-inc", "false",
  "prepare update form when the load/store increment is a loop invariant non-const value">;
defm : ValueField<"ppc-preinc-prep-max-vars", "unsigned", "3",
  "Potential PHI threshold per loop for PPC loop prep of update form">;

// PPCMachineFunctionInfo.cpp
defm : BoolField<"ppc-non-volatile-cr", "true",
  "Enable the use of non-volatile CR register fields">;

// PPCMachineScheduler.cpp
defm : BoolField<"ppc-postra-bias-addi", "true",
  "Enable scheduling addi instruction as early as possible post ra">;
defm : BoolField<"ppc-sched-addi-load", "true",
  "Enable scheduling addi instruction before load for ppc">;

// PPCMIPeephole.cpp
defm : BoolField<"ppc-convert-rr-to-ri", "true",
  "Convert eligible reg+reg instructions to reg+imm">;
defm : BoolField<"ppc-eliminate-signext", "true",
  "enable elimination of sign-extensions">;
defm : BoolField<"ppc-eliminate-zeroext", "true",
  "enable elimination of zero-extensions">;
defm : BoolField<"ppc-opt-conditional-trap", "false",
  "enable optimization of conditional traps">;
defm : BoolField<"ppc-reg-to-imm-fixed-point", "true",
  "Iterate to a fixed point when attempting to convert reg-reg instructions to reg-imm">;

// PPCPreEmitPeephole.cpp
defm : BoolField<"ppc-late-peephole", "true",
  "Run pre-emit peephole optimizations">;
defm : BoolField<"ppc-pcrel-linker-opt", "true",
  "enable PC Relative linker optimization">;
defm : OptionalValueField<"ppc-set-dscr", "uint64_t",
  "Set the Data Stream Control Register">;

// PPCRegisterInfo.cpp
defm : BoolField<"ppc-always-use-base-pointer", "false",
  "Force the use of a base pointer in every function">;
defm : BoolField<"ppc-enable-gpr-to-vsr-spills", "false",
  "Enable spills from gpr to vsr rather than stack">;
defm : ValueField<"ppc-max-crbit-spill-dist", "unsigned", "100",
  "Maximum search distance for definition of CR bit spill on ppc">;
// Copies/moves of physical accumulators are expensive operations that should be
// avoided whenever possible. MMA instructions are meant to be used in
// performance-sensitive computational kernels. This option is provided, at
// least for the time being, to give the user a tool to detect this expensive
// operation and either rework their code or report a compiler bug if that turns
// out to be the cause.
let Condition = AssertsOnly in
defm : BoolField<"ppc-report-acc-moves", "false",
  "Emit information about accumulator register spills and copies">;
defm : BoolField<"ppc-stack-ptr-caller-preserved", "true",
  "Consider R1 caller preserved so stack saves of caller preserved registers can be LICM candidates">;
defm : BoolField<"ppc-use-base-pointer", "true",
  "Enable use of a base pointer for complex stack frames">;

// PPCTargetMachine.cpp
defm : BoolField<"ppc-branch-coalesce", "false",
  "enable coalescing of duplicate branches for PPC">;
defm : BoolField<"ppc-ctr-loops", "true",
  "Enable CTR loops for PPC">;
defm : BoolField<"ppc-extra-toc-reg-deps", "true",
  "Add extra TOC register dependencies">;
defm : BoolField<"ppc-gep-opt", "true",
  "Enable optimizations on complex GEPs">;
defm : OptionalBoolField<"ppc-global-merge",
  "Enable the global merge pass">;
defm : ValueField<"ppc-global-merge-max-offset", "unsigned", "0x7fff",
  "Maximum global merge offset">;
defm : BoolField<"ppc-instr-form-prep", "true",
  "Enable PPC loop instr form prep">;
defm : BoolField<"ppc-machine-combiner", "true",
  "Enable the machine combiner pass">;
defm : BoolField<"ppc-peephole", "true",
  "Enable machine peepholes for PPC">;
defm : OptionalBoolField<"ppc-prefetching",
  "enable software prefetching on PPC">;
defm : BoolField<"ppc-reduce-cr-logicals", "true",
  "Expand eligible cr-logical binary ops to branches">;
defm : BoolField<"ppc-scalar-mass", "false",
  "Enable lowering math functions to their corresponding MASS (scalar) entries">;
defm : BoolField<"ppc-schedule-vsx-fma-mutation-early", "false",
  "Schedule VSX FMA instruction mutation early">;
defm : BoolField<"ppc-vsx-swap-removal", "true",
  "Enable VSX Swap Removal for PPC">;

// PPCTargetTransformInfo.cpp
defm : BoolField<"ppc-constant-hoisting", "true",
  "enable constant hoisting on PPC">;
defm : BoolField<"ppc-enable-coldcc", "false",
  "Enable using coldcc calling conv for cold internal functions">;
defm : BoolField<"ppc-evl", "false",
  "Allow EVL type vp.load/vp.store">;
defm : BoolField<"ppc-lsr-no-insns-cost", "false",
  "Do not add instruction count to lsr cost model">;
// The latency of mtctr is only justified if there are more than 4 comparisons
// that will be removed as a result.
defm : ValueField<"ppc-min-ctr-loop-threshold", "unsigned", "4",
  "Loops with a constant trip count smaller than this value will not use the count register">;
defm : BoolField<"ppc-pwr9-evl", "false",
  "Allow vp.load and vp.store for pwr9">;
defm : BoolField<"ppc-vec-mask-cost", "true",
  "add masking cost for i1 vectors">;

// PPCVSXFMAMutate.cpp
// Temporarily disable FMA mutation by default, since it doesn't handle
// cross-basic-block intervals well. See:
// http://lists.llvm.org/pipermail/llvm-dev/2016-February/095669.html
// http://reviews.llvm.org/D17087
defm : BoolField<"ppc-vsx-fma-mutation", "false",
  "Enable VSX FMA instruction mutation">;
