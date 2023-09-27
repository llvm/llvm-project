//===-- RISCVGIsel.td - RISCV GlobalISel Patterns ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains patterns that are relevant to GlobalISel, including
/// GIComplexOperandMatcher definitions for equivalent SelectionDAG
/// ComplexPatterns.
//
//===----------------------------------------------------------------------===//

include "RISCV.td"
include "RISCVCombine.td"

def simm12Plus1 : ImmLeaf<XLenVT, [{
    return (isInt<12>(Imm) && Imm != -2048) || Imm == 2048;}]>;

def GINegImm : GICustomOperandRenderer<"renderNegImm">,
  GISDNodeXFormEquiv<NegImm>;

// FIXME: This is labelled as handling 's32', however the ComplexPattern it
// refers to handles both i32 and i64 based on the HwMode. Currently this LLT
// parameter appears to be ignored so this pattern works for both, however we
// should add a LowLevelTypeByHwMode, and use that to define our XLenLLT instead
// here.
def ShiftMaskGI :
    GIComplexOperandMatcher<s32, "selectShiftMask">,
    GIComplexPatternEquiv<shiftMaskXLen>;

// FIXME: Canonicalize (sub X, C) -> (add X, -C) earlier.
def : Pat<(XLenVT (sub GPR:$rs1, simm12Plus1:$imm)),
          (ADDI GPR:$rs1, (NegImm simm12Plus1:$imm))>;

let Predicates = [IsRV64] in {
def : Pat<(i32 (add GPR:$rs1, GPR:$rs2)), (ADDW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (sub GPR:$rs1, GPR:$rs2)), (SUBW GPR:$rs1, GPR:$rs2)>;

def : Pat<(i32 (shl GPR:$rs1, (i32 GPR:$rs2))), (SLLW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (sra GPR:$rs1, (i32 GPR:$rs2))), (SRAW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (srl GPR:$rs1, (i32 GPR:$rs2))), (SRLW GPR:$rs1, GPR:$rs2)>;

def: Pat<(i64 (sext i32:$rs)), (ADDIW GPR:$rs, 0)>;
}

let Predicates = [HasStdExtMOrZmmul, IsRV64] in {
def : Pat<(i32 (mul GPR:$rs1, GPR:$rs2)), (MULW GPR:$rs1, GPR:$rs2)>;
}

let Predicates = [HasStdExtM, IsRV64] in {
def : Pat<(i32 (sdiv GPR:$rs1, GPR:$rs2)), (DIVW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (srem GPR:$rs1, GPR:$rs2)), (REMW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (udiv GPR:$rs1, GPR:$rs2)), (DIVUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (urem GPR:$rs1, GPR:$rs2)), (REMUW GPR:$rs1, GPR:$rs2)>;
}

let Predicates = [HasStdExtZba, IsRV64] in
def : Pat<(i64 (zext i32:$rs)), (ADD_UW GPR:$rs, (XLenVT X0))>;

let Predicates = [IsRV64, NotHasStdExtZba] in
def: Pat<(i64 (zext i32:$rs)), (SRLI (SLLI GPR:$rs, 32), 32)>;
