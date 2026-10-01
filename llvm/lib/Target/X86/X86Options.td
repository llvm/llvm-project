//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def X86Options : OptionsStruct<prefix = "x86-">;

defm : ValueField<"branch-hint-probability-threshold", "unsigned", "50",
  "The probability threshold of enabling branch hint">;
defm : BoolField<"disable-spill-fusing", "false",
  "Disable fusing of spill code into instructions">;
defm : BoolField<"disable-x86-domain-reassignment", "false",
  "X86: Disable Virtual Register Reassignment">;
defm : BoolField<"disable-x86-lea-opt", "false",
  "X86: Disable LEA optimizations">;
defm : BoolField<"enable-branch-hint", "false",
  "Enable branch hint">;
defm : BoolField<"enable-x86-scalar-amx", "false",
  "X86: enable AMX scalarizition">;
defm : BoolField<"fixup-byte-word-insts", "true",
  "Change byte and word instructions to larger sizes">;
defm : BoolField<"mul-constant-optimization", "true",
  "Replace 'mul x, Const' with more effective instructions like SHIFT, LEA, etc">;
defm : BoolField<"no-x86-call-frame-opt", "false",
  "Avoid optimizing x86 call frames for size">;
defm : ValueField<"partial-reg-update-clearance", "unsigned", "64",
  "Clearance between two register writes for inserting XOR to avoid partial register update">;
defm : BoolField<"print-failed-fuse-candidates", "false",
  "Print instructions that the allocator wants to fuse, but the X86 backend currently can't">;
defm : BoolField<"remat-pic-stub-load", "false",
  "Re-materialize load from stub in PIC mode">;
defm : ValueField<"undef-reg-clearance", "unsigned", "128",
  "How many idle instructions we would like before certain undef register reads">;
defm : BoolField<"x86-and-imm-shrink", "true",
  "Enable setting constant bits to reduce size of mask immediates">;
defm : ValueField<"x86-br-merging-base-cost", "int", "2",
  "Sets the cost threshold for when multiple conditionals will be merged into one branch versus be split in multiple branches. "
  "Merging conditionals saves branches at the cost of additional instructions. This value sets the instruction cost limit, below which conditionals will be merged, and above which conditionals will be split. Set to -1 to never merge branches">;
defm : ValueField<"x86-br-merging-ccmp-bias", "int", "6",
  "Increases 'x86-br-merging-base-cost' in cases that the target supports conditional compare instructions">;
defm : ValueField<"x86-br-merging-likely-bias", "int", "0",
  "Increases 'x86-br-merging-base-cost' in cases that it is likely that all conditionals will be executed. "
  "For example for merging the conditionals (a == b && c > d), if its known that a == b is likely, then it is likely that if the conditionals are split both sides will be executed, so it may be desirable to increase the instruction cost threshold. Set to -1 to never merge likely branches">;
defm : ValueField<"x86-br-merging-unlikely-bias", "int", "-1",
  "Decreases 'x86-br-merging-base-cost' in cases that it is unlikely that all conditionals will be executed. "
  "For example for merging the conditionals (a == b && c > d), if its known that a == b is unlikely, then it is unlikely that if the conditionals are split both sides will be executed, so it may be desirable to decrease the instruction cost threshold. Set to -1 to never merge unlikely branches">;
defm : BoolField<"x86-cmov-converter", "true",
  "Enable the X86 cmov-to-branch optimization">;
defm : BoolField<"x86-cmov-converter-force-all", "false",
  "Convert all cmovs to branches">;
defm : BoolField<"x86-cmov-converter-force-mem-operand", "true",
  "Convert cmovs to branches whenever they have memory operands">;
defm : ValueField<"x86-cmov-converter-threshold", "unsigned", "4",
  "Minimum gain per loop (in cycles) threshold">;
defm : BoolField<"x86-disable-avoid-SFB", "false",
  "X86: Disable Store Forwarding Blocks fixup">;
defm : BoolField<"x86-disable-regalloc-hints-for-ndd", "false",
  "Disable two address hints for register allocation">;
// Temporary option to control early if-conversion for x86 while adding machine
// models.
defm : BoolField<"x86-early-ifcvt", "false",
  "Enable early if-conversion on X86">;
defm : BoolField<"x86-enable-apx-for-relocation", "false",
  "Enable APX features (EGPR, NDD and NF) for instructions with relocations on x86-64 ELF">;
defm : ValueField<"x86-experimental-pref-innermost-loop-alignment", "std::optional<int>", "std::nullopt",
  "Sets the preferable loop alignment for experiments (as log2 bytes) for innermost loops only. "
  "If specified, this option overrides alignment set by x86-experimental-pref-loop-alignment">;
defm : BoolField<"x86-indirect-branch-tracking", "false",
  "Enable X86 indirect branch tracking pass">;
defm : BoolField<"x86-lvi-load-dot", "false",
  "For each function, emit a dot graph depicting potential LVI gadgets">;
defm : BoolField<"x86-lvi-load-dot-only", "false",
  "For each function, emit a dot graph depicting potential LVI gadgets, and do not insert any fences">;
defm : BoolField<"x86-lvi-load-dot-verify", "false",
  "For each function, emit a dot graph to stdout depicting potential LVI gadgets, used for testing purposes only">;
defm : BoolField<"x86-lvi-load-no-cbranch", "false",
  "Don't treat conditional branches as disclosure gadgets. "
  "This may improve performance, at the cost of security">;
defm : ValueField<"x86-lvi-load-opt-plugin", "StringRef", "",
  "Specify a plugin to optimize LFENCE insertion">;
defm : BoolField<"x86-machine-combiner", "true",
  "Enable the machine combiner pass">;
defm : ValueField<"x86-max-nf-conversions-for-cmp-reuse", "unsigned", "6",
  "Maximum number of NF conversions allowed to reuse EFLAGS from a producer dominating a multi-predecessor block">;
defm : BoolField<"x86-promote-anyext-load", "true",
  "Enable promoting aligned anyext load to wider load">;
defm : BoolField<"x86-seses-enable-without-lvi-cfi", "false",
  "Force enable speculative execution side effect suppression. "
  "(Note: User must pass -mlvi-cfi in order to mitigate indirect branches and returns.)">;
defm : BoolField<"x86-seses-omit-branch-lfences", "false",
  "Omit all lfences before branch instructions">;
defm : BoolField<"x86-seses-one-lfence-per-bb", "false",
  "Omit all lfences other than the first to be placed in a basic block">;
defm : BoolField<"x86-seses-only-lfence-non-const", "false",
  "Only lfence before groups of terminators where at least one branch instruction has an input to the addressing mode that is a register other than %rip">;
defm : ValueField<"x86-setjmp-csr-warning-threshold", "unsigned", "50",
  "Basic block count threshold for emitting a warning about callee-saved registers reserved due to setjmp">;
defm : ValueField<"x86-sfb-inspection-limit", "unsigned", "20",
  "X86: Number of instructions backward to inspect for store forwarding blocks">;
defm : BoolField<"x86-slh-fence-call-and-ret", "false",
  "Use a full speculation fence to harden both call and ret edges rather than a lighter weight mitigation">;
defm : BoolField<"x86-slh-indirect", "true",
  "Harden indirect calls and jumps against using speculatively stored attacker controlled addresses. "
  "This is designed to mitigate Spectre v1.2 style attacks">;
defm : BoolField<"x86-slh-ip", "true",
  "Harden interprocedurally by passing our state in and out of functions in the high bits of the stack pointer">;
defm : BoolField<"x86-slh-lfence", "false",
  "Use LFENCE along each conditional edge to harden against speculative loads rather than conditional movs and poisoned pointers">;
defm : BoolField<"x86-slh-loads", "true",
  "Sanitize loads from memory. "
  "When disable, no significant security is provided">;
defm : BoolField<"x86-slh-post-load", "true",
  "Harden the value loaded *after* it is loaded by flushing the loaded bits to 1. "
  "This is hard to do in general but can be done easily for GPRs">;
defm : BoolField<"x86-speculative-load-hardening", "false",
  "Force enable speculative load hardening">;
defm : BoolField<"x86-tile-ra", "true",
  "Enable the tile register allocation pass">;
defm : BoolField<"x86-use-base-pointer", "true",
  "Enable use of a base pointer for complex stack frames">;
defm : BoolField<"x86-use-fsrm-for-memcpy", "false",
  "Use fast short rep mov in memcpy lowering">;
defm : BoolField<"x86-use-vzeroupper", "true",
  "Minimize AVX to SSE transition penalty">;
defm : BoolField<"x86-widen-shift", "true",
  "Replace narrow shifts with wider shifts">;
defm : ValueField<"x86-wineh-unwindv2-force-mode", "std::optional<unsigned>", "std::nullopt",
  "Overwrites the Unwind v2 mode for testing purposes">;
// This threshold is for the *approximate* number of instructions, see the
// comment in runAnalysisOnFuncOrFunclet in X86WinEHUnwindV2.cpp for more
// details.
defm : ValueField<"x86-wineh-unwindv2-instruction-count-threshold", "unsigned", "600",
  "Maximum number of (approximate) instructions before splitting into a new unwind info">;
defm : ValueField<"x86-wineh-unwindv2-unwind-codes-threshold", "unsigned", "UINT8_MAX",
  "Maximum number of unwind codes before splitting into a new unwind info">;
defm : ValueField<"x86-wineh-unwindv3-instr-avg-size", "unsigned", "7",
  "Average size of an instruction. "
  "This value is used in determining split points for chained unwinder info">;
