//===-- RISCVInstrPredicates.td - Instruction Predicates ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instruction predicates.
//
//===----------------------------------------------------------------------===//

// This predicate is true when the rs2 operand of vlse or vsse is x0, false
// otherwise.
def VLDSX0Pred : MCSchedPredicate<CheckRegOperand<3, X0>>;

// This scheduling predicate is true when subtarget feature TuneHasSingleElementVecFP64
// is enabled.
def SingleElementVecFP64SchedPred : FeatureSchedPredicate<TuneHasSingleElementVecFP64>;

// Returns true if this is the sext.w pattern, addiw rd, rs1, 0.
def isSEXT_W
    : TIIPredicate<"isSEXT_W",
                   MCReturnStatement<CheckAll<[
                     CheckOpcode<[ADDIW]>,
                     CheckIsRegOperand<1>,
                     CheckIsImmOperand<2>,
                     CheckImmOperand<2, 0>
                   ]>>>;

// Returns true if this is the zext.w pattern, adduw rd, rs1, x0.
def isZEXT_W
    : TIIPredicate<"isZEXT_W",
                   MCReturnStatement<CheckAll<[
                     CheckOpcode<[ADD_UW]>,
                     CheckIsRegOperand<1>,
                     CheckIsRegOperand<2>,
                     CheckRegOperand<2, X0>
                   ]>>>;

// Returns true if this is the zext.b pattern, andi rd, rs1, 255.
def isZEXT_B
    : TIIPredicate<"isZEXT_B",
                   MCReturnStatement<CheckAll<[
                     CheckOpcode<[ANDI]>,
                     CheckIsRegOperand<1>,
                     CheckIsImmOperand<2>,
                     CheckImmOperand<2, 255>
                   ]>>>;

def isSelectPseudo
    : TIIPredicate<"isSelectPseudo",
                   MCReturnStatement<
                     CheckOpcode<[
                       Select_GPR_Using_CC_GPR,
                       Select_GPR_Using_CC_Imm5_Zibi,
                       Select_GPR_Using_CC_SImm5_CV,
                       Select_GPRNoX0_Using_CC_SImm5NonZero_QC,
                       Select_GPRNoX0_Using_CC_UImm5NonZero_QC,
                       Select_GPRNoX0_Using_CC_SImm16NonZero_QC,
                       Select_GPRNoX0_Using_CC_UImm16NonZero_QC,
                       Select_GPR_Using_CC_UImmLog2XLen_NDS,
                       Select_GPR_Using_CC_UImm7_NDS,
                       Select_FPR16_Using_CC_GPR,
                       Select_FPR16INX_Using_CC_GPR,
                       Select_FPR32_Using_CC_GPR,
                       Select_FPR32INX_Using_CC_GPR,
                       Select_FPR64_Using_CC_GPR,
                       Select_FPR64INX_Using_CC_GPR,
                       Select_FPR64IN32X_Using_CC_GPR
                     ]>>>;

// Returns true if this is a vector configuration instruction.
def isVectorConfigInstr
    : TIIPredicate<"isVectorConfigInstr",
                   MCReturnStatement<
                     CheckOpcode<[
                       PseudoVSETVLI,
                       PseudoVSETVLIX0,
                       PseudoVSETVLIX0X0,
                       PseudoVSETIVLI
                     ]>>>;

// Return true if this is 'vsetvli x0, x0, vtype' which preserves
// VL and only sets VTYPE.
def isVLPreservingConfig
    : TIIPredicate<"isVLPreservingConfig",
                   MCReturnStatement<CheckOpcode<[PseudoVSETVLIX0X0]>>>;

def isFloatScalarMoveOrScalarSplatInstr
    : TIIPredicate<"isFloatScalarMoveOrScalarSplatInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !instances<Pseudo>("^PseudoVFMV_S_F.*"),
                      !instances<Pseudo>("^PseudoVFMV_V_F.*")
                     ])>>>;

def isScalarExtractInstr
    : TIIPredicate<"isScalarExtractInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !instances<Pseudo>("^PseudoVMV_X_S.*"),
                      !instances<Pseudo>("^PseudoVFMV_F.*_S.*")
                     ])>>>;

def isVExtractInstr
    : TIIPredicate<"isVExtractInstr",
                   MCReturnStatement<
                     CheckOpcode<
                      !instances<Instruction>("^PseudoRI_VEXTRACT.*")>>>;

def isScalarInsertInstr
    : TIIPredicate<"isScalarInsertInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !instances<Pseudo>("^PseudoVMV_S_X.*"),
                      !instances<Pseudo>("^PseudoVFMV_S_F.*")
                     ])>>>;

def isScalarSplatInstr
    : TIIPredicate<"isScalarSplatInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !instances<Pseudo>("^PseudoVMV_V_I.*"),
                      !instances<Pseudo>("^PseudoVMV_V_X.*"),
                      !instances<Pseudo>("^PseudoVFMV_V_F.*")
                     ])>>>;

def isVSlideInstr
    : TIIPredicate<"isVSlideInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !instances<Pseudo>("^PseudoVSLIDEDOWN_VX.*"),
                      !instances<Pseudo>("^PseudoVSLIDEDOWN_VI.*"),
                      !instances<Pseudo>("^PseudoVSLIDEUP_VX.*"),
                      !instances<Pseudo>("^PseudoVSLIDEUP_VI.*")
                     ])>>>;

def isFaultOnlyFirstLoad
    : TIIPredicate<"isFaultOnlyFirstLoad",
                    MCReturnStatement<
                      CheckOpcode<
                       !instances<Pseudo>(
                          "^PseudoVL(SEG[2-8])?E(8|16|32|64)FF_V.*")>>>;

def isNonZeroLoadImmediate
    : TIIPredicate<"isNonZeroLoadImmediate",
                   MCReturnStatement<CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckIsRegOperand<1>,
                     CheckRegOperand<1, X0>,
                     CheckIsImmOperand<2>,
                     CheckNot<CheckImmOperand<2, 0>>
                   ]>>>;

def isLPAD
    : TIIPredicate<"isLPAD",
                   MCReturnStatement<CheckAll<[
                     CheckOpcode<[AUIPC]>,
                     CheckIsRegOperand<0>,
                     CheckRegOperand<0, X0>,
                   ]>>>;

def ignoresVXRM
    : TIIPredicate<"ignoresVXRM",
                   MCOpcodeSwitchStatement<
                     [MCOpcodeSwitchCase<
                        !listflatten([
                          !instances<Pseudo>("^PseudoVNCLIP_WI.*"),
                          !instances<Pseudo>("^PseudoVNCLIPU_WI.*")
                        ]),
                        MCReturnStatement<CheckImmOperand<3, 0>>>],
                      MCReturnStatement<FalsePred>>>;
