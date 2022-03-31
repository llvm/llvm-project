//===-- LoongArch.td - Describe the LoongArch Target -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// LoongArch subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

// LoongArch is divided into two versions, the 32-bit version (LA32) and the
// 64-bit version (LA64).
def Feature64Bit
    : SubtargetFeature<"64bit", "HasLA64", "true",
                       "LA64 Basic Integer and Privilege Instruction Set">;
def IsLA64
    : Predicate<"Subtarget->is64Bit()">,
      AssemblerPredicate<(all_of Feature64Bit),
                         "LA64 Basic Integer and Privilege Instruction Set">;
def IsLA32
    : Predicate<"!Subtarget->is64Bit()">,
      AssemblerPredicate<(all_of(not Feature64Bit)),
                         "LA32 Basic Integer and Privilege Instruction Set">;

defvar LA32 = DefaultMode;
def LA64 : HwMode<"+64bit">;

// Single Precision floating point
def FeatureBasicF : SubtargetFeature<"f", "HasBasicF", "true",
                                     "'F' (Single-Precision Floating-Point)">;
def HasBasicF : Predicate<"Subtarget->hasBasicF()">,
                AssemblerPredicate<(all_of FeatureBasicF),
                                   "'F' (Single-Precision Floating-Point)">;

// Double Precision floating point
def FeatureBasicD : SubtargetFeature<"d", "HasBasicD", "true",
                                     "'D' (Double-Precision Floating-Point)",
                                     [FeatureBasicF]>;
def HasBasicD : Predicate<"Subtarget->hasBasicD()">,
                AssemblerPredicate<(all_of FeatureBasicD),
                                   "'D' (Double-Precision Floating-Point)">;

// Loongson SIMD eXtension (LSX)
def FeatureExtLSX
    : SubtargetFeature<"lsx", "HasExtLSX", "true",
                       "'LSX' (Loongson SIMD Extension)", [FeatureBasicD]>;
def HasExtLSX : Predicate<"Subtarget->hasExtLSX()">,
                AssemblerPredicate<(all_of FeatureExtLSX),
                                   "'LSX' (Loongson SIMD Extension)">;

// Loongson Advanced SIMD eXtension (LASX)
def FeatureExtLASX
    : SubtargetFeature<"lasx", "HasExtLASX", "true",
                       "'LASX' (Loongson Advanced SIMD Extension)",
                       [FeatureExtLSX]>;
def HasExtLASX
    : Predicate<"Subtarget->hasExtLASX()">,
      AssemblerPredicate<(all_of FeatureExtLASX),
                         "'LASX' (Loongson Advanced SIMD Extension)">;

// Loongson VirtualiZation (LVZ)
def FeatureExtLVZ
    : SubtargetFeature<"lvz", "HasExtLVZ", "true",
                       "'LVZ' (Loongson Virtualization Extension)">;
def HasExtLVZ : Predicate<"Subtarget->hasExtLVZ()">,
                AssemblerPredicate<(all_of FeatureExtLVZ),
                                   "'LVZ' (Loongson Virtualization Extension)">;

// Loongson Binary Translation (LBT)
def FeatureExtLBT
    : SubtargetFeature<"lbt", "HasExtLBT", "true",
                       "'LBT' (Loongson Binary Translation Extension)">;
def HasExtLBT
    : Predicate<"Subtarget->hasExtLBT()">,
      AssemblerPredicate<(all_of FeatureExtLBT),
                         "'LBT' (Loongson Binary Translation Extension)">;

//===----------------------------------------------------------------------===//
// Registers, instruction descriptions ...
//===----------------------------------------------------------------------===//

include "LoongArchRegisterInfo.td"
include "LoongArchCallingConv.td"
include "LoongArchInstrInfo.td"

//===----------------------------------------------------------------------===//
// LoongArch processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic-la32", NoSchedModel, []>;
def : ProcessorModel<"generic-la64", NoSchedModel, [Feature64Bit]>;

def : ProcessorModel<"la464", NoSchedModel, [Feature64Bit,
                                             FeatureExtLASX,
                                             FeatureExtLVZ,
                                             FeatureExtLBT]>;

//===----------------------------------------------------------------------===//
// Define the LoongArch target.
//===----------------------------------------------------------------------===//

def LoongArchInstrInfo : InstrInfo {
  // guess mayLoad, mayStore, and hasSideEffects
  // This option is a temporary migration help. It will go away.
  let guessInstructionProperties = 1;
}

def LoongArchAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def LoongArchAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  // Recognize hard coded registers.
  string RegisterPrefix = "$";
}

def LoongArchAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def LoongArch : Target {
  let InstructionSet = LoongArchInstrInfo;
  let AssemblyParsers = [LoongArchAsmParser];
  let AssemblyParserVariants = [LoongArchAsmParserVariant];
  let AssemblyWriters = [LoongArchAsmWriter];
  let AllowRegisterRenaming = 1;
}
