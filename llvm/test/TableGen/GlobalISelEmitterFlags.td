// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def InstThreeOperands : I<(outs GPR32:$dst), (ins GPR32:$cond, GPR32:$src,GPR32:$src2), []>;

class SrlPF<bit IC>: PatFrag<
  (ops node:$PATFRAG_Src0, node:$src1),
  (srl $PATFRAG_Src0, $src1)>, GISelFlags {
  let GIIgnoreCopies = IC;
}

// GIIgnoreCopies on Pattern
//  MIs[1] should be using IgnoreCopies variants.
let GIIgnoreCopies = 1 in
def : Pat<
  (i32 (sub (mul i32:$src0, i32:$src0), i32:$src1)),
  (InstThreeOperands GPR32:$src0, GPR32:$src1, GPR32:$src1)
>, GISelFlags;

// GIIgnoreCopies set on "root" PatFrag.
//  MIs[1] and MIs[2] should be using IgnoreCopies variants.
def : Pat<
  (i32 (SrlPF<1> (shl (mul i32:$src0, i32:$src0), i32:$src1), i32:$src0)),
  (InstThreeOperands GPR32:$src0, GPR32:$src1, GPR32:$src1)
>;

// GIIgnoreCopies set on "root" PatFrag, but a children PatFrag forces it back to zero.
//    MIs[1] should be using IgnoreCopies variants.
//    MIs[2] should NOT be using them.
def : Pat<
  (i32 (SrlPF<1> (SrlPF<0> (add i32:$src0, i32:$src0), i32:$src1), i32:$src0)),
  (InstThreeOperands GPR32:$src0, GPR32:$src1, GPR32:$src1)
>;

// CHECK: GIM_Try
// CHECK:    GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK:    GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_LSHR
// CHECK:    GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK:    GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD
// CHECK:    GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1
// CHECK:    GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1
// CHECK:    // (srl:{ *:[i32] } (srl:{ *:[i32] } (add:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src0), i32:{ *:[i32] }:$src1), i32:{ *:[i32] }:$src0)
// CHECK:   GIR_Done
// CHECK: GIM_Try
// CHECK:   GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK:   GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SHL
// CHECK:   GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK:   GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_MUL
// CHECK:   GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1
// CHECK:   GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1
// CHECK:   // (srl:{ *:[i32] } (shl:{ *:[i32] } (mul:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src0), i32:{ *:[i32] }:$src1), i32:{ *:[i32] }:$src0)
// CHECK:   GIR_Done
// CHECK: GIM_Try
// CHECK:   GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK:   GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL
// CHECK:   GIM_CheckIsSameOperandIgnoreCopies, /*MI*/1, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1
// CHECK:   // (sub:{ *:[i32] } (mul:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src0), i32:{ *:[i32] }:$src1)  =>  (InstThreeOperands:{ *:[i32] } GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src1)
// CHECK:   GIR_Done