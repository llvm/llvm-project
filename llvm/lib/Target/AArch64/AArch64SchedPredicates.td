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

// Check the extension type in arithmetic instructions.
let FunctionMapper = "AArch64_AM::getArithExtendType" in {
  def CheckExtUXTB                      : CheckImmOperand_s<3, "AArch64_AM::UXTB">;
  def CheckExtUXTH                      : CheckImmOperand_s<3, "AArch64_AM::UXTH">;
  def CheckExtUXTW                      : CheckImmOperand_s<3, "AArch64_AM::UXTW">;
  def CheckExtUXTX                      : CheckImmOperand_s<3, "AArch64_AM::UXTX">;
  def CheckExtSXTB                      : CheckImmOperand_s<3, "AArch64_AM::SXTB">;
  def CheckExtSXTH                      : CheckImmOperand_s<3, "AArch64_AM::SXTH">;
  def CheckExtSXTW                      : CheckImmOperand_s<3, "AArch64_AM::SXTW">;
  def CheckExtSXTX                      : CheckImmOperand_s<3, "AArch64_AM::SXTX">;
}

// Check for shifting in extended arithmetic instructions.
foreach I = {0-3} in {
  let FunctionMapper = "AArch64_AM::getArithShiftValue" in
  def CheckExtBy#I                      : CheckImmOperand<3, I>;
}

// Check the extension type in the register offset addressing mode.
let FunctionMapper = "AArch64_AM::getMemExtendType" in {
  def CheckMemExtUXTW                   : CheckImmOperand_s<3, "AArch64_AM::UXTW">;
  def CheckMemExtLSL                    : CheckImmOperand_s<3, "AArch64_AM::UXTX">;
  def CheckMemExtSXTW                   : CheckImmOperand_s<3, "AArch64_AM::SXTW">;
  def CheckMemExtSXTX                   : CheckImmOperand_s<3, "AArch64_AM::SXTX">;
}

// Check for scaling in the register offset addressing mode.
let FunctionMapper = "AArch64_AM::getMemDoShift" in
def CheckMemScaled                      : CheckImmOperandSimple<3>;

// Check the shifting type in arithmetic and logic instructions.
let FunctionMapper = "AArch64_AM::getShiftType" in {
  def CheckShiftLSL                : CheckImmOperand_s<3, "AArch64_AM::LSL">;
  def CheckShiftLSR                : CheckImmOperand_s<3, "AArch64_AM::LSR">;
  def CheckShiftASR                : CheckImmOperand_s<3, "AArch64_AM::ASR">;
  def CheckShiftROR                : CheckImmOperand_s<3, "AArch64_AM::ROR">;
  def CheckShiftMSL                : CheckImmOperand_s<3, "AArch64_AM::MSL">;
}

// Check for shifting in arithmetic and logic instructions.
foreach I = {0-3, 8} in {
  let FunctionMapper = "AArch64_AM::getShiftValue" in
  def CheckShiftBy#I        : CheckImmOperand<3, I>;
}

// Generic predicates.

// Identify whether an instruction is the 64-bit NEON form based on its result.
def CheckDForm             : CheckAll<[CheckIsRegOperand<0>,
                                       CheckAny<[CheckRegOperand<0, D0>,
                                                 CheckRegOperand<0, D1>,
                                                 CheckRegOperand<0, D2>,
                                                 CheckRegOperand<0, D3>,
                                                 CheckRegOperand<0, D4>,
                                                 CheckRegOperand<0, D5>,
                                                 CheckRegOperand<0, D6>,
                                                 CheckRegOperand<0, D7>,
                                                 CheckRegOperand<0, D8>,
                                                 CheckRegOperand<0, D9>,
                                                 CheckRegOperand<0, D10>,
                                                 CheckRegOperand<0, D11>,
                                                 CheckRegOperand<0, D12>,
                                                 CheckRegOperand<0, D13>,
                                                 CheckRegOperand<0, D14>,
                                                 CheckRegOperand<0, D15>,
                                                 CheckRegOperand<0, D16>,
                                                 CheckRegOperand<0, D17>,
                                                 CheckRegOperand<0, D18>,
                                                 CheckRegOperand<0, D19>,
                                                 CheckRegOperand<0, D20>,
                                                 CheckRegOperand<0, D21>,
                                                 CheckRegOperand<0, D22>,
                                                 CheckRegOperand<0, D23>,
                                                 CheckRegOperand<0, D24>,
                                                 CheckRegOperand<0, D25>,
                                                 CheckRegOperand<0, D26>,
                                                 CheckRegOperand<0, D27>,
                                                 CheckRegOperand<0, D28>,
                                                 CheckRegOperand<0, D29>,
                                                 CheckRegOperand<0, D30>,
                                                 CheckRegOperand<0, D31>]>]>;

// Identify whether an instruction is the 128-bit NEON form based on its result.
def CheckQForm             : CheckAll<[CheckIsRegOperand<0>,
                                       CheckAny<[CheckRegOperand<0, Q0>,
                                                 CheckRegOperand<0, Q1>,
                                                 CheckRegOperand<0, Q2>,
                                                 CheckRegOperand<0, Q3>,
                                                 CheckRegOperand<0, Q4>,
                                                 CheckRegOperand<0, Q5>,
                                                 CheckRegOperand<0, Q6>,
                                                 CheckRegOperand<0, Q7>,
                                                 CheckRegOperand<0, Q8>,
                                                 CheckRegOperand<0, Q9>,
                                                 CheckRegOperand<0, Q10>,
                                                 CheckRegOperand<0, Q11>,
                                                 CheckRegOperand<0, Q12>,
                                                 CheckRegOperand<0, Q13>,
                                                 CheckRegOperand<0, Q14>,
                                                 CheckRegOperand<0, Q15>,
                                                 CheckRegOperand<0, Q16>,
                                                 CheckRegOperand<0, Q17>,
                                                 CheckRegOperand<0, Q18>,
                                                 CheckRegOperand<0, Q19>,
                                                 CheckRegOperand<0, Q20>,
                                                 CheckRegOperand<0, Q21>,
                                                 CheckRegOperand<0, Q22>,
                                                 CheckRegOperand<0, Q23>,
                                                 CheckRegOperand<0, Q24>,
                                                 CheckRegOperand<0, Q25>,
                                                 CheckRegOperand<0, Q26>,
                                                 CheckRegOperand<0, Q27>,
                                                 CheckRegOperand<0, Q28>,
                                                 CheckRegOperand<0, Q29>,
                                                 CheckRegOperand<0, Q30>,
                                                 CheckRegOperand<0, Q31>]>]>;

// Identify arithmetic instructions with extend.
def IsArithExtOp           : CheckOpcode<[ADDWrx, ADDXrx, ADDSWrx, ADDSXrx,
                                          SUBWrx, SUBXrx, SUBSWrx, SUBSXrx,
                                          ADDXrx64, ADDSXrx64,
                                          SUBXrx64, SUBSXrx64]>;

// Identify arithmetic immediate instructions.
def IsArithImmOp           : CheckOpcode<[ADDWri, ADDXri, ADDSWri, ADDSXri,
                                          SUBWri, SUBXri, SUBSWri, SUBSXri]>;

// Identify arithmetic instructions with shift.
def IsArithShiftOp         : CheckOpcode<[ADDWrs, ADDXrs, ADDSWrs, ADDSXrs,
                                          SUBWrs, SUBXrs, SUBSWrs, SUBSXrs]>;

// Identify arithmetic instructions without shift.
def IsArithUnshiftOp       : CheckOpcode<[ADDWrr, ADDXrr, ADDSWrr, ADDSXrr,
                                          SUBWrr, SUBXrr, SUBSWrr, SUBSXrr]>;

// Identify logic immediate instructions.
def IsLogicImmOp           : CheckOpcode<[ANDWri, ANDXri,
                                          EORWri, EORXri,
                                          ORRWri, ORRXri]>;

// Identify logic instructions with shift.
def IsLogicShiftOp         : CheckOpcode<[ANDWrs, ANDXrs, ANDSWrs, ANDSXrs,
                                          BICWrs, BICXrs, BICSWrs, BICSXrs,
                                          EONWrs, EONXrs,
                                          EORWrs, EORXrs,
                                          ORNWrs, ORNXrs,
                                          ORRWrs, ORRXrs]>;

// Identify logic instructions without shift.
def IsLogicUnshiftOp       : CheckOpcode<[ANDWrr, ANDXrr, ANDSWrr, ANDSXrr,
                                          BICWrr, BICXrr, BICSWrr, BICSXrr,
                                          EONWrr, EONXrr,
                                          EORWrr, EORXrr,
                                          ORNWrr, ORNXrr,
                                          ORRWrr, ORRXrr]>;

// Identify arithmetic and logic immediate instructions.
def IsArithLogicImmOp      : CheckOpcode<!listconcat(IsArithImmOp.ValidOpcodes,
                                                     IsLogicImmOp.ValidOpcodes)>;

// Identify arithmetic and logic instructions with shift.
def IsArithLogicShiftOp    : CheckOpcode<!listconcat(IsArithShiftOp.ValidOpcodes,
                                                     IsLogicShiftOp.ValidOpcodes)>;

// Identify arithmetic and logic instructions without shift.
def IsArithLogicUnshiftOp  : CheckOpcode<!listconcat(IsArithUnshiftOp.ValidOpcodes,
                                                     IsLogicUnshiftOp.ValidOpcodes)>;

// Identify whether an instruction whose result is a long vector
// operates on the upper half of the input registers.
def IsLongVectorUpperOp    : CheckOpcode<[FCVTLv8i16, FCVTLv4i32,
                                          FCVTNv8i16, FCVTNv4i32,
                                          FCVTXNv4f32,
                                          PMULLv16i8, PMULLv2i64,
                                          RADDHNv8i16_v16i8, RADDHNv4i32_v8i16, RADDHNv2i64_v4i32,
                                          RSHRNv16i8_shift, RSHRNv8i16_shift, RSHRNv4i32_shift,
                                          RSUBHNv8i16_v16i8, RSUBHNv4i32_v8i16, RSUBHNv2i64_v4i32,
                                          SABALv16i8_v8i16, SABALv8i16_v4i32, SABALv4i32_v2i64,
                                          SABDLv16i8_v8i16, SABDLv8i16_v4i32, SABDLv4i32_v2i64,
                                          SADDLv16i8_v8i16, SADDLv8i16_v4i32, SADDLv4i32_v2i64,
                                          SADDWv16i8_v8i16, SADDWv8i16_v4i32, SADDWv4i32_v2i64,
                                          SHLLv16i8, SHLLv8i16, SHLLv4i32,
                                          SHRNv16i8_shift, SHRNv8i16_shift, SHRNv4i32_shift,
                                          SMLALv16i8_v8i16, SMLALv8i16_v4i32, SMLALv4i32_v2i64,
                                          SMLALv8i16_indexed, SMLALv4i32_indexed,
                                          SMLSLv16i8_v8i16, SMLSLv8i16_v4i32, SMLSLv4i32_v2i64,
                                          SMLSLv8i16_indexed, SMLSLv4i32_indexed,
                                          SMULLv16i8_v8i16, SMULLv8i16_v4i32, SMULLv4i32_v2i64,
                                          SMULLv8i16_indexed, SMULLv4i32_indexed,
                                          SQDMLALv8i16_v4i32, SQDMLALv4i32_v2i64,
                                          SQDMLALv8i16_indexed, SQDMLALv4i32_indexed,
                                          SQDMLSLv8i16_v4i32, SQDMLSLv4i32_v2i64,
                                          SQDMLSLv8i16_indexed, SQDMLSLv4i32_indexed,
                                          SQDMULLv8i16_v4i32, SQDMULLv4i32_v2i64,
                                          SQDMULLv8i16_indexed, SQDMULLv4i32_indexed,
                                          SQRSHRNv16i8_shift, SQRSHRNv8i16_shift, SQRSHRNv4i32_shift,
                                          SQRSHRUNv16i8_shift, SQRSHRUNv8i16_shift, SQRSHRUNv4i32_shift,
                                          SQSHRNv16i8_shift, SQSHRNv8i16_shift, SQSHRNv4i32_shift,
                                          SQSHRUNv16i8_shift, SQSHRUNv8i16_shift, SQSHRUNv4i32_shift,
                                          SQXTNv16i8, SQXTNv8i16, SQXTNv4i32,
                                          SQXTUNv16i8, SQXTUNv8i16, SQXTUNv4i32,
                                          SSHLLv16i8_shift, SSHLLv8i16_shift, SSHLLv4i32_shift,
                                          SSUBLv16i8_v8i16, SSUBLv8i16_v4i32, SSUBLv4i32_v2i64,
                                          SSUBWv16i8_v8i16, SSUBWv8i16_v4i32, SSUBWv4i32_v2i64,
                                          UABALv16i8_v8i16, UABALv8i16_v4i32, UABALv4i32_v2i64,
                                          UABDLv16i8_v8i16, UABDLv8i16_v4i32, UABDLv4i32_v2i64,
                                          UADDLv16i8_v8i16, UADDLv8i16_v4i32, UADDLv4i32_v2i64,
                                          UADDWv16i8_v8i16, UADDWv8i16_v4i32, UADDWv4i32_v2i64,
                                          UMLALv16i8_v8i16, UMLALv8i16_v4i32, UMLALv4i32_v2i64,
                                          UMLALv8i16_indexed, UMLALv4i32_indexed,
                                          UMLSLv16i8_v8i16, UMLSLv8i16_v4i32, UMLSLv4i32_v2i64,
                                          UMLSLv8i16_indexed, UMLSLv4i32_indexed,
                                          UMULLv16i8_v8i16, UMULLv8i16_v4i32, UMULLv4i32_v2i64,
                                          UMULLv8i16_indexed, UMULLv4i32_indexed,
                                          UQSHRNv16i8_shift, UQSHRNv8i16_shift, UQSHRNv4i32_shift,
                                          UQXTNv16i8, UQXTNv8i16, UQXTNv4i32,
                                          USHLLv16i8_shift, USHLLv8i16_shift, USHLLv4i32_shift,
                                          USUBLv16i8_v8i16, USUBLv8i16_v4i32, USUBLv4i32_v2i64,
                                          USUBWv16i8_v8i16, USUBWv8i16_v4i32, USUBWv4i32_v2i64,
                                          XTNv16i8, XTNv8i16, XTNv4i32]>;

// Identify whether an instruction is a load
// using the register offset addressing mode.
def IsLoadRegOffsetOp      : CheckOpcode<[PRFMroW, PRFMroX,
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
def IsStoreRegOffsetOp     : CheckOpcode<[STRBBroW, STRBBroX,
                                          STRHHroW, STRHHroX,
                                          STRWroW, STRWroX,
                                          STRXroW, STRXroX,
                                          STRBroW, STRBroX,
                                          STRHroW, STRHroX,
                                          STRSroW, STRSroX,
                                          STRDroW, STRDroX]>;

// Identify whether an instruction is a load or
// store using the register offset addressing mode.
def IsLoadStoreRegOffsetOp : CheckOpcode<!listconcat(IsLoadRegOffsetOp.ValidOpcodes,
                                                     IsStoreRegOffsetOp.ValidOpcodes)>;

// Target predicates.

// Identify an instruction that effectively transfers a register to another.
def IsCopyIdiomFn     : TIIPredicate<"isCopyIdiom",
                                     MCOpcodeSwitchStatement<
                                       [// MOV {Rd, SP}, {SP, Rn} =>
                                        // ADD {Rd, SP}, {SP, Rn}, #0
                                        MCOpcodeSwitchCase<
                                          [ADDWri, ADDXri],
                                          MCReturnStatement<
                                            CheckAll<
                                              [CheckIsRegOperand<0>,
                                               CheckIsRegOperand<1>,
                                               CheckAny<
                                                 [CheckRegOperand<0, WSP>,
                                                  CheckRegOperand<0, SP>,
                                                  CheckRegOperand<1, WSP>,
                                                  CheckRegOperand<1, SP>]>,
                                               CheckZeroOperand<2>]>>>,
                                        // MOV Rd, Rm =>
                                        // ORR Rd, ZR, Rm, LSL #0
                                        MCOpcodeSwitchCase<
                                          [ORRWrs, ORRXrs],
                                          MCReturnStatement<
                                            CheckAll<
                                              [CheckIsRegOperand<1>,
                                               CheckIsRegOperand<2>,
                                               CheckAny<
                                                 [CheckRegOperand<1, WZR>,
                                                  CheckRegOperand<1, XZR>,
                                                  CheckRegOperand<2, WZR>,
                                                  CheckRegOperand<2, XZR>]>,
                                               CheckShiftBy0]>>>],
                                       MCReturnStatement<FalsePred>>>;
def IsCopyIdiomPred   : MCSchedPredicate<IsCopyIdiomFn>;

// Identify arithmetic instructions with an extended register.
def RegExtendedFn     : TIIPredicate<"hasExtendedReg",
                                     MCOpcodeSwitchStatement<
                                       [MCOpcodeSwitchCase<
                                         IsArithExtOp.ValidOpcodes,
                                         MCReturnStatement<
                                           CheckNot<CheckZeroOperand<3>>>>],
                                       MCReturnStatement<FalsePred>>>;
def RegExtendedPred   : MCSchedPredicate<RegExtendedFn>;

// Identify arithmetic and logic instructions with a shifted register.
def RegShiftedFn      : TIIPredicate<"hasShiftedReg",
                                     MCOpcodeSwitchStatement<
                                       [MCOpcodeSwitchCase<
                                          IsArithLogicShiftOp.ValidOpcodes,
                                          MCReturnStatement<
                                            CheckNot<CheckZeroOperand<3>>>>],
                                       MCReturnStatement<FalsePred>>>;
def RegShiftedPred    : MCSchedPredicate<RegShiftedFn>;

// Identify a load or store using the register offset addressing mode
// with an extended or scaled register.
def ScaledIdxFn       : TIIPredicate<"isScaledAddr",
                                     MCOpcodeSwitchStatement<
                                       [MCOpcodeSwitchCase<
                                          IsLoadStoreRegOffsetOp.ValidOpcodes,
                                          MCReturnStatement<
                                            CheckAny<[CheckNot<CheckMemExtLSL>,
                                                      CheckMemScaled]>>>],
                                       MCReturnStatement<FalsePred>>>;
def ScaledIdxPred     : MCSchedPredicate<ScaledIdxFn>;

// Identify an instruction that effectively resets a FP register to zero.
def IsZeroFPIdiomFn   : TIIPredicate<"isZeroFPIdiom",
                                     MCOpcodeSwitchStatement<
                                       [// MOVI Vd, #0
                                        MCOpcodeSwitchCase<
                                          [MOVIv8b_ns, MOVIv16b_ns,
                                           MOVID, MOVIv2d_ns],
                                          MCReturnStatement<CheckZeroOperand<1>>>,
                                        // MOVI Vd, #0, LSL #0
                                        MCOpcodeSwitchCase<
                                          [MOVIv4i16, MOVIv8i16,
                                           MOVIv2i32, MOVIv4i32],
                                          MCReturnStatement<
                                            CheckAll<
                                              [CheckZeroOperand<1>,
                                               CheckZeroOperand<2>]>>>],
                                       MCReturnStatement<FalsePred>>>;
def IsZeroFPIdiomPred : MCSchedPredicate<IsZeroFPIdiomFn>;

// Identify an instruction that effectively resets a GP register to zero.
def IsZeroIdiomFn     : TIIPredicate<"isZeroIdiom",
                                    MCOpcodeSwitchStatement<
                                      [// ORR Rd, ZR, #0
                                       MCOpcodeSwitchCase<
                                         [ORRWri, ORRXri],
                                         MCReturnStatement<
                                           CheckAll<
                                             [CheckIsRegOperand<1>,
                                              CheckAny<
                                                [CheckRegOperand<1, WZR>,
                                                 CheckRegOperand<1, XZR>]>,
                                              CheckZeroOperand<2>]>>>],
                                      MCReturnStatement<FalsePred>>>;
def IsZeroIdiomPred   : MCSchedPredicate<IsZeroIdiomFn>;
