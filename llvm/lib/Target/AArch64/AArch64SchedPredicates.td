//===- AArch64SchedPredicates.td - AArch64 Sched Preds -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines scheduling predicate definitions that are used by the
// AArch64 subtargets.
//
//===----------------------------------------------------------------------===//

// Function mappers.

// Check the extension type in the register offset addressing mode.
let FunctionMapper = "AArch64_AM::getMemExtendType" in {
  def CheckMemExtUXTW : CheckImmOperand_s<3, "AArch64_AM::UXTW">;
  def CheckMemExtLSL  : CheckImmOperand_s<3, "AArch64_AM::UXTX">;
  def CheckMemExtSXTW : CheckImmOperand_s<3, "AArch64_AM::SXTW">;
  def CheckMemExtSXTX : CheckImmOperand_s<3, "AArch64_AM::SXTX">;
}

// Check for scaling in the register offset addressing mode.
let FunctionMapper = "AArch64_AM::getMemDoShift" in
def CheckMemScaled  : CheckImmOperandSimple<3>;

// Generic predicates.

// Identify arithmetic instructions with extend.
def IsArithExtPred        : CheckOpcode<[ADDWrx, ADDXrx, ADDXrx64, ADDSWrx, ADDSXrx, ADDSXrx64,
                                         SUBWrx, SUBXrx, SUBXrx64, SUBSWrx, SUBSXrx, SUBSXrx64]>;

// Identify arithmetic instructions with shift.
def IsArithShiftPred      : CheckOpcode<[ADDWrs, ADDXrs, ADDSWrs, ADDSXrs,
                                         SUBWrs, SUBXrs, SUBSWrs, SUBSXrs]>;

// Identify logic instructions with shift.
def IsLogicShiftPred      : CheckOpcode<[ANDWrs, ANDXrs, ANDSWrs, ANDSXrs,
                                         BICWrs, BICXrs, BICSWrs, BICSXrs,
                                         EONWrs, EONXrs,
                                         EORWrs, EORXrs,
                                         ORNWrs, ORNXrs,
                                         ORRWrs, ORRXrs]>;

// Identify arithmetic and logic instructions with shift.
def IsArithLogicShiftPred : CheckAny<[IsArithShiftPred, IsLogicShiftPred]>;

// Identify whether an instruction is a load
// using the register offset addressing mode.
def IsLoadRegOffsetPred   : CheckOpcode<[PRFMroW, PRFMroX,
                                         LDRBBroW, LDRBBroX,
                                         LDRSBWroW, LDRSBWroX, LDRSBXroW, LDRSBXroX,
                                         LDRHHroW, LDRHHroX,
                                         LDRSHWroW, LDRSHWroX, LDRSHXroW, LDRSHXroX,
                                         LDRWroW, LDRWroX,
                                         LDRSWroW, LDRSWroX,
                                         LDRXroW, LDRXroX,
                                         LDRBroW, LDRBroX,
                                         LDRHroW, LDRHroX,
                                         LDRSroW, LDRSroX,
                                         LDRDroW, LDRDroX]>;

// Identify whether an instruction is a load
// using the register offset addressing mode.
def IsStoreRegOffsetPred  : CheckOpcode<[STRBBroW, STRBBroX,
                                         STRHHroW, STRHHroX,
                                         STRWroW, STRWroX,
                                         STRXroW, STRXroX,
                                         STRBroW, STRBroX,
                                         STRHroW, STRHroX,
                                         STRSroW, STRSroX,
                                         STRDroW, STRDroX]>;

// Target predicates.

// Identify arithmetic instructions with an extended register.
def RegExtendedFn   : TIIPredicate<"hasExtendedReg",
                                   MCOpcodeSwitchStatement<
                                     [MCOpcodeSwitchCase<
                                       IsArithExtPred.ValidOpcodes,
                                       MCReturnStatement<CheckNot<CheckZeroOperand<3>>>>],
                                     MCReturnStatement<FalsePred>>>;
def RegExtendedPred : MCSchedPredicate<RegExtendedFn>;

// Identify arithmetic and logic instructions with a shifted register.
def RegShiftedFn    : TIIPredicate<"hasShiftedReg",
                                   MCOpcodeSwitchStatement<
                                     [MCOpcodeSwitchCase<
                                        !listconcat(IsArithShiftPred.ValidOpcodes,
                                                    IsLogicShiftPred.ValidOpcodes),
                                        MCReturnStatement<CheckNot<CheckZeroOperand<3>>>>],
                                   MCReturnStatement<FalsePred>>>;
def RegShiftedPred  : MCSchedPredicate<RegShiftedFn>;

// Identify a load or store using the register offset addressing mode
// with an extended or scaled register.
def ScaledIdxFn     : TIIPredicate<"isScaledAddr",
                                   MCOpcodeSwitchStatement<
                                     [MCOpcodeSwitchCase<
                                        !listconcat(IsLoadRegOffsetPred.ValidOpcodes,
                                                    IsStoreRegOffsetPred.ValidOpcodes),
                                        MCReturnStatement<
                                          CheckAny<[CheckNot<CheckMemExtLSL>,
                                                    CheckMemScaled]>>>],
                                     MCReturnStatement<FalsePred>>>;
def ScaledIdxPred   : MCSchedPredicate<ScaledIdxFn>;
