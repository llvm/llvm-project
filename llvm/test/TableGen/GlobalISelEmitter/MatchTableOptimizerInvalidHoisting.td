// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=true -I %p/../../../include -I %p/../Common | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def InstThreeOperands : I<(outs GPR32:$dst), (ins GPR32:$src0, GPR32:$src1, GPR32:$src2), []>;

class ThreeOpFrag<SDPatternOperator op1, SDPatternOperator op2> : PatFrag<
  (ops node:$x, node:$y, node:$z),
  (op2 (op1 node:$x, node:$y), node:$z),
  [{
    return Operands[0] && Operands[1] && Operands[2];
  }]> {
  let PredicateCodeUsesOperands = 1;
  let GISelPredicateCode = [{
    return Operands[0] && Operands[1] && Operands[2];
  }];
}

def ptradd_commutative : PatFrags<(ops node:$src0, node:$src1),
  [(ptradd node:$src0, node:$src1), (ptradd node:$src1, node:$src0)]>;

// ptradd_commutative has two PatFrags, therefore there are two ways how the
// below pattern could match. Both require checking the C++ predicate, but that
// check cannot be hoisted because it relies on recorded operands, which differ
// between the PatFrags. This is inspired by a similar construct in the AMDGPU
// backend.

// CHECK: GIM_Try, /*On fail goto*//*Label 1*/
// CHECK: GIM_RecordNamedOperand
// CHECK: GIM_RecordNamedOperand
// CHECK: GIM_RecordNamedOperand
// CHECK: GIM_CheckCxxInsnPredicate
// CHECK: // Label 1
// CHECK: GIM_Try, /*On fail goto*//*Label 2*/
// CHECK: GIM_RecordNamedOperand
// CHECK: GIM_RecordNamedOperand
// CHECK: GIM_RecordNamedOperand
// CHECK: GIM_CheckCxxInsnPredicate
// CHECK: // Label 2
def : Pat <(i32 (ThreeOpFrag<shl, ptradd_commutative> GPR32:$src0, GPR32:$src1, GPR32:$src2)),
  (InstThreeOperands GPR32:$src0, GPR32:$src1, GPR32:$src2)> ;

