//===- AArch64SchedPredNeoverse.td - AArch64 Sched Preds -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines scheduling predicate definitions that are used by the
// AArch64 Neoverse processors.
//
//===----------------------------------------------------------------------===//

// Auxiliary predicates.

// Check for LSL shift <= 4
def NeoverseCheapLSL : MCSchedPredicate<
                         CheckAll<
                           [CheckShiftLSL,
                            CheckAny<
                              [CheckShiftBy0,
                               CheckShiftBy1,
                               CheckShiftBy2,
                               CheckShiftBy3,
                               CheckShiftBy4]>]>>;

// Check for LSL shift == 0
def NeoverseNoLSL : MCSchedPredicate<
                      CheckAll<[CheckShiftLSL,
                                CheckShiftBy0]>>;

// Check if the fourth operand of an instruction is WZR or XZR
def NeoverseReg3IsZero : MCSchedPredicate<
                           CheckAll<
                             [CheckIsRegOperand<3>,
                              CheckAny<
                                [CheckRegOperand<3, WZR>,
                                 CheckRegOperand<3, XZR>]>]>>;

// Identify LDR/STR H/Q-form scaled (and potentially extended) FP instructions
def NeoverseHQForm : MCSchedPredicate<
                       CheckAll<[
                         CheckAny<[CheckHForm, CheckQForm]>,
                         CheckImmOperand<4, 1>]>>;

// Check if <Pd> == <Pg>
def NeoversePdIsPgFn : TIIPredicate<
                         "isNeoversePdSameAsPg",
                         MCOpcodeSwitchStatement<
                           [MCOpcodeSwitchCase<[BRKA_PPmP, BRKB_PPmP],
                             MCReturnStatement<CheckSameRegOperand<1, 2>>>],
                           MCReturnStatement<CheckSameRegOperand<0, 1>>>>;
def NeoversePdIsPg : MCSchedPredicate<NeoversePdIsPgFn>;

// Check if SVE INC/DEC (scalar), ALL, {1, 2, 4}
def NeoverseCheapIncDec : MCSchedPredicate<
                            CheckAll<[CheckOpcode<[
                                        INCB_XPiI, INCH_XPiI,
                                        INCW_XPiI, INCD_XPiI,
                                        DECB_XPiI, DECH_XPiI,
                                        DECW_XPiI, DECD_XPiI]>,
                                      CheckImmOperand<2, 31>,
                                      CheckAny<[
                                        CheckImmOperand<3, 1>,
                                        CheckImmOperand<3, 2>,
                                        CheckImmOperand<3, 4>]>]>>;
