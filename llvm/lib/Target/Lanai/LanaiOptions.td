//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def LanaiOptions : OptionsStruct<prefix = "lanai-">;

// LanaiDelaySlotFiller.cpp
defm : BoolField<"lanai-nop-delay-filler", "false", "Fill Lanai delay slots with NOPs">;

// LanaiISelLowering.cpp
// Limit on number of instructions the lowered multiplication may have before a
// call to the library function should be generated instead. The threshold is
// currently set to 14 as this was the smallest threshold that resulted in all
// constant multiplications being lowered. A threshold of 5 covered all cases
// except for one multiplication which required 14. mulsi3 requires 16
// instructions (including the prologue and epilogue but excluding instructions
// at call site). Until we can inline mulsi3, generating at most 14 instructions
// will be faster than invoking mulsi3.
defm : ValueField<"lanai-constant-mul-threshold", "int", "14", "Maximum number of instruction to generate when lowering constant multiplication instead of calling library function">;

// LanaiMemAluCombiner.cpp
defm : BoolField<"lanai-enable-mem-alu-combiner", "true", "Combine ALU and memory operators">;

// LanaiTargetObjectFile.cpp
defm : ValueField<"lanai-ssection-threshold", "unsigned", "0", "Small data and bss section threshold size">;

// TODO: Remove the old spellings after 2027-04.
def : AliasOf<"disable-lanai-mem-alu-combiner", "lanai-enable-mem-alu-combiner=false">;
