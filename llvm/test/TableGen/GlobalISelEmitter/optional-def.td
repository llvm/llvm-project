// RUN: llvm-tblgen -gen-global-isel  -warn-on-skipped-patterns \
// RUN:   -I %p/../../../include -I %p/../Common %s 2> %t | FileCheck %s
// RUN: FileCheck -DFILE=%s -check-prefix=ERR %s < %t

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def cc_out : OptionalDefOperand<i32, (ops GPR8), (ops (i8 zero_reg))>;
def s_cc_out : OptionalDefOperand<i32, (ops GPR8, FPR32), (ops (i8 B0), F0)>;

// CHECK-LABEL: // (add:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)  =>  (tst2:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst2),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[rd]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/0, GIMT_Encode2(MyTarget::B0), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define | RegState::Dead),
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/0, GIMT_Encode2(MyTarget::F0), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define | RegState::Dead),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/1, // rs1
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/2, // rs2
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 1,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: // (imm:{ *:[i32] }):$imm  =>  (tst1:{ *:[i32] } (imm:{ *:[i32] }):$imm)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst1),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[rd]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/0, GIMT_Encode2(MyTarget::NoRegister), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define | RegState::Dead),
// CHECK-NEXT: GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // imm
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

def tst1 : I<(outs GPR32:$rd, cc_out:$s), (ins i32imm:$imm),
             [(set GPR32:$rd, imm:$imm)]>;

def tst2 : I<(outs GPR32:$rd, s_cc_out:$s), (ins GPR32:$rs1, GPR32:$rs2),
             [(set GPR32:$rd, (add i32:$rs1, i32:$rs2))]>;

// TODO: There should be more tests, but any attempt to write something
//   more complex results in tablegen crashing somewhere in
//   TreePatternNode::UpdateNodeType.


def not_leaf : OptionalDefOperand<i32, (ops GPR8), (ops (i8 imm))>;
def not_rec  : OptionalDefOperand<i32, (ops GPR8), (ops (i8 0))>;
def not_reg  : OptionalDefOperand<i32, (ops GPR8), (ops GPR8)>;

// ERR: [[#@LINE+1]]:5: warning: Skipped pattern: optional def is not a leaf
def tst_not_leaf : I<(outs GPR32:$rd, not_leaf:$s), (ins i32imm:$imm),
                     [(set GPR32:$rd, imm:$imm)]>;

// ERR: [[#@LINE+1]]:5: warning: Skipped pattern: optional def is not a record
def tst_not_rec : I<(outs GPR32:$rd, not_rec:$s), (ins i32imm:$imm),
                    [(set GPR32:$rd, imm:$imm)]>;

// ERR: [[#@LINE+1]]:5: warning: Skipped pattern: optional def is not a register
def tst_not_reg : I<(outs GPR32:$rd, not_reg:$s), (ins i32imm:$imm),
                    [(set GPR32:$rd, imm:$imm)]>;
