//===- AArch64SchedPredExynos.td - AArch64 Sched Preds -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines scheduling predicate definitions that are used by the
// AArch64 Exynos processors.
//
//===----------------------------------------------------------------------===//

// Auxiliary predicates.

// Check the shift in arithmetic and logic instructions.
def ExynosCheckShift : CheckAny<[CheckShiftBy0,
                                 CheckAll<
                                   [CheckShiftLSL,
                                    CheckAny<
                                      [CheckShiftBy1,
                                       CheckShiftBy2,
                                       CheckShiftBy3]>]>]>;

// Exynos predicates.

// Identify BLR specifying the LR register as the indirect target register.
def ExynosBranchLinkLRPred : MCSchedPredicate<
                               CheckAll<[CheckOpcode<[BLR]>,
                                         CheckRegOperand<0, LR>]>>;

// Identify arithmetic and logic instructions without or with limited extension.
def ExynosExtFn   : TIIPredicate<
                      "isExynosExtFast",
                      MCOpcodeSwitchStatement<
                        [MCOpcodeSwitchCase<
                           IsArithExtOp.ValidOpcodes,
                           MCReturnStatement<
                             CheckAny<[CheckExtBy0,
                                       CheckAll<
                                         [CheckAny<
                                           [CheckExtUXTW,
                                            CheckExtUXTX]>,
                                          CheckAny<
                                            [CheckExtBy1,
                                             CheckExtBy2,
                                             CheckExtBy3]>]>]>>>],
                        MCReturnStatement<FalsePred>>>;
def ExynosExtPred : MCSchedPredicate<ExynosExtFn>;

// Identify a load or store using the register offset addressing mode
// with a scaled non-extended register.
def ExynosScaledIdxFn   : TIIPredicate<"isExynosScaledAddr",
                                       MCOpcodeSwitchStatement<
                                         [MCOpcodeSwitchCase<
                                            IsLoadStoreRegOffsetOp.ValidOpcodes,
                                            MCReturnStatement<
                                              CheckAny<
                                                [CheckMemExtSXTW,
                                                 CheckMemExtUXTW,
                                                 CheckMemScaled]>>>],
                                         MCReturnStatement<FalsePred>>>;
def ExynosScaledIdxPred : MCSchedPredicate<ExynosScaledIdxFn>;

// Identify FP instructions.
def ExynosFPPred : MCSchedPredicate<CheckAny<[CheckDForm, CheckQForm]>>;

// Identify whether an instruction whose result is a long vector
// operates on the upper half of the input registers.
def ExynosLongVectorUpperFn   : TIIPredicate<
                                  "isExynosLongVectorUpper",
                                  MCOpcodeSwitchStatement<
                                  [MCOpcodeSwitchCase<
                                    IsLongVectorUpperOp.ValidOpcodes,
                                    MCReturnStatement<TruePred>>],
                                  MCReturnStatement<FalsePred>>>;
def ExynosLongVectorUpperPred : MCSchedPredicate<ExynosLongVectorUpperFn>;

// Identify 128-bit NEON instructions.
def ExynosQFormPred : MCSchedPredicate<CheckQForm>;

// Identify instructions that reset a register efficiently.
def ExynosResetFn   : TIIPredicate<
                        "isExynosResetFast",
                        MCOpcodeSwitchStatement<
                          [MCOpcodeSwitchCase<
                             [ADR, ADRP,
                              MOVNWi, MOVNXi,
                              MOVZWi, MOVZXi],
                             MCReturnStatement<TruePred>>,
                           MCOpcodeSwitchCase<
                             [ORRWri, ORRXri],
                             MCReturnStatement<
                               CheckAll<
                                 [CheckIsRegOperand<1>,
                                  CheckAny<
                                    [CheckRegOperand<1, WZR>,
                                     CheckRegOperand<1, XZR>]>]>>>],
                          MCReturnStatement<
                            CheckAny<
                              [IsCopyIdiomFn,
                               IsZeroFPIdiomFn]>>>>;
def ExynosResetPred : MCSchedPredicate<ExynosResetFn>;

// Identify EXTR as the alias for ROR (immediate).
def ExynosRotateRightImmPred : MCSchedPredicate<
                                 CheckAll<[CheckOpcode<[EXTRWrri, EXTRXrri]>,
                                           CheckSameRegOperand<1, 2>]>>;

// Identify arithmetic and logic instructions with limited shift.
def ExynosShiftFn   : TIIPredicate<
                        "isExynosShiftFast",
                        MCOpcodeSwitchStatement<
                          [MCOpcodeSwitchCase<
                             IsArithLogicShiftOp.ValidOpcodes,
                             MCReturnStatement<ExynosCheckShift>>],
                          MCReturnStatement<FalsePred>>>;
def ExynosShiftPred : MCSchedPredicate<ExynosShiftFn>;

// Identify more arithmetic and logic instructions with limited shift.
def ExynosShiftExFn   : TIIPredicate<
                          "isExynosShiftExFast",
                          MCOpcodeSwitchStatement<
                            [MCOpcodeSwitchCase<
                               IsArithLogicShiftOp.ValidOpcodes,
                               MCReturnStatement<
                                 CheckAny<
                                   [ExynosCheckShift,
                                    CheckAll<
                                     [CheckShiftLSL,
                                      CheckShiftBy8]>]>>>],
                            MCReturnStatement<FalsePred>>>;
def ExynosShiftExPred : MCSchedPredicate<ExynosShiftExFn>;

// Identify arithmetic and logic immediate instructions.
def ExynosCheapFn : TIIPredicate<
                      "isExynosCheapAsMove",
                      MCOpcodeSwitchStatement<
                        [MCOpcodeSwitchCase<
                           IsArithLogicImmOp.ValidOpcodes,
                           MCReturnStatement<TruePred>>],
                        MCReturnStatement<
                          CheckAny<
                            [ExynosExtFn, ExynosResetFn, ExynosShiftFn]>>>>;
