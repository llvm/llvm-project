//===-- AMDGPUGIsel.td - AMDGPU GlobalISel Patterns---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This files contains patterns that should only be used by GlobalISel.  For
// example patterns for V_* instructions that have S_* equivalents.
// SelectionDAG does not support selecting V_* instructions.
//===----------------------------------------------------------------------===//

include "AMDGPU.td"

def sd_vsrc0 : ComplexPattern<i32, 1, "">;
def gi_vsrc0 :
    GIComplexOperandMatcher<s32, "selectVSRC0">,
    GIComplexPatternEquiv<sd_vsrc0>;

def gi_vop3mods0 :
    GIComplexOperandMatcher<s32, "selectVOP3Mods0">,
    GIComplexPatternEquiv<VOP3Mods0>;

class GISelSop2Pat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src0_vt SReg_32:$src0), (src1_vt SReg_32:$src1))),
  (inst src0_vt:$src0, src1_vt:$src1)
>;

class GISelVop2Pat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src0_vt (sd_vsrc0 src0_vt:$src0)), (src1_vt VGPR_32:$src1))),
  (inst src0_vt:$src0, src1_vt:$src1)
>;

def : GISelSop2Pat <or, S_OR_B32, i32>;
def : GISelVop2Pat <or, V_OR_B32_e32, i32>;
