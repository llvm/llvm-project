//===-- RISCV.td - Describe the RISCV Target Machine -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// RISC-V subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

def FeatureStdExtM
    : SubtargetFeature<"m", "HasStdExtM", "true",
                       "'M' (Integer Multiplication and Division)">;
def HasStdExtM : Predicate<"Subtarget->hasStdExtM()">,
                           AssemblerPredicate<"FeatureStdExtM">;

def FeatureStdExtA
    : SubtargetFeature<"a", "HasStdExtA", "true",
                       "'A' (Atomic Instructions)">;
def HasStdExtA : Predicate<"Subtarget->hasStdExtA()">,
                           AssemblerPredicate<"FeatureStdExtA">;

def Feature64Bit
    : SubtargetFeature<"64bit", "HasRV64", "true", "Implements RV64">;

def RV64           : HwMode<"+64bit">;
def RV32           : HwMode<"-64bit">;

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "RISCVRegisterInfo.td"
include "RISCVCallingConv.td"
include "RISCVInstrInfo.td"

//===----------------------------------------------------------------------===//
// RISC-V processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic-rv32", NoSchedModel, []>;

def : ProcessorModel<"generic-rv64", NoSchedModel, [Feature64Bit]>;

//===----------------------------------------------------------------------===//
// Define the RISC-V target.
//===----------------------------------------------------------------------===//

def RISCVInstrInfo : InstrInfo {
  let guessInstructionProperties = 0;
}

def RISCVAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
}

def RISCV : Target {
  let InstructionSet = RISCVInstrInfo;
  let AssemblyParsers = [RISCVAsmParser];
}
