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

// Returns true if this is the zext.b pattern, andi rd, rs1, 255.
def isSelectPseudo
    : TIIPredicate<"isSelectPseudo",
                   MCOpcodeSwitchStatement<
                     [MCOpcodeSwitchCase<
                        [Select_GPR_Using_CC_GPR,
                         Select_GPR_Using_CC_Imm,
                         Select_FPR16_Using_CC_GPR,
                         Select_FPR16INX_Using_CC_GPR,
                         Select_FPR32_Using_CC_GPR,
                         Select_FPR32INX_Using_CC_GPR,
                         Select_FPR64_Using_CC_GPR,
                         Select_FPR64INX_Using_CC_GPR,
                         Select_FPR64IN32X_Using_CC_GPR
                        ],
                        MCReturnStatement<TruePred>>],
                      MCReturnStatement<FalsePred>>>;

// Returns true if this is a vector configuration instruction.
def isVectorConfigInstr
    : TIIPredicate<"isVectorConfigInstr",
                   MCReturnStatement<
                     CheckOpcode<[
                       PseudoVSETVLI,
                       PseudoVSETVLIX0,
                       PseudoVSETIVLI
                     ]>>>;

// Return true if this is 'vsetvli x0, x0, vtype' which preserves
// VL and only sets VTYPE.
def isVLPreservingConfig
    : TIIPredicate<"isVLPreservingConfig",
                   MCReturnStatement<
                     CheckAll<[
                       CheckOpcode<[PseudoVSETVLIX0]>,
                       CheckRegOperand<0, X0>
                     ]>>>;

def isFloatScalarMoveOrScalarSplatInstr
    : TIIPredicate<"isFloatScalarMoveOrScalarSplatInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !select<Instruction>(".*PseudoVFMV_S_F.*"),
                      !select<Instruction>(".*PseudoVFMV_V_F.*")
                     ])>>>;

def isScalarExtractInstr
    : TIIPredicate<"isScalarExtractInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !select<Instruction>(".*PseudoVMV_X_S.*"),
                      !select<Instruction>(".*PseudoVFMV_F.*_S.*")
                     ])>>>;

def isScalarInsertInstr
    : TIIPredicate<"isScalarInsertInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !select<Instruction>(".*PseudoVMV_S_X.*"),
                      !select<Instruction>(".*PseudoVFMV_S_F.*")
                     ])>>>;

def isScalarSplatInstr
    : TIIPredicate<"isScalarSplatInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !select<Instruction>(".*PseudoVMV_V_I.*"),
                      !select<Instruction>(".*PseudoVMV_V_X.*"),
                      !select<Instruction>(".*PseudoVFMV_V_F.*")
                     ])>>>;

def isVSlideInstr
    : TIIPredicate<"isVSlideInstr",
                   MCReturnStatement<
                     CheckOpcode<!listflatten([
                      !select<Instruction>(".*PseudoVSLIDEDOWN_VX.*"),
                      !select<Instruction>(".*PseudoVSLIDEDOWN_VI.*"),
                      !select<Instruction>(".*PseudoVSLIDEUP_VX.*"),
                      !select<Instruction>(".*PseudoVSLIDEUP_VI.*")
                     ])>>>;

def isNonZeroLoadImmediate
    : TIIPredicate<"isNonZeroLoadImmediate",
                   MCReturnStatement<CheckAll<[
                     CheckOpcode<[ADDI]>,
                     CheckIsRegOperand<1>,
                     CheckRegOperand<1, X0>,
                     CheckIsImmOperand<2>,
                     CheckNot<CheckImmOperand<2, 0>>
                   ]>>>;

def ignoresVXRM
    : TIIPredicate<"ignoresVXRM",
                   MCOpcodeSwitchStatement<
                     [MCOpcodeSwitchCase<
                        !listflatten([
                          !select<Instruction>(".*PseudoVNCLIP_WI.*"),
                          !select<Instruction>(".*PseudoVNCLIPU_WI.*")
                        ]),
                        MCReturnStatement<CheckImmOperand<3, 0>>>],
                      MCReturnStatement<FalsePred>>>;
