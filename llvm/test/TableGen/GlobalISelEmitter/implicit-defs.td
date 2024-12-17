// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false \
// RUN:   -I %p/../../../include -I %p/../Common %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let Defs = [R0, B0] in
def tst1 : I<(outs), (ins), [(set R0, (i32 42))]>;

let Defs = [R0, R1] in
def tst2 : I<(outs GPR32:$rd), (ins GPR32:$rs1, GPR32:$rs2),
             [(set GPR32:$rd, R1, (sdivrem i32:$rs1, i32:$rs2))]>;

let Defs = [R0, R1] in
def tst3 : I<(outs), (ins GPR32:$rs1, GPR32:$rs2),
             [(set R1, R0, (udivrem i32:$rs1, i32:$rs2))]>;

let Defs = [R0] in
def tst4 : I<(outs GPR32:$rd), (ins GPR32:$rs), []>;

def : Pat<(sdivrem i32:$rs, 42), (tst4 (tst4 $rs))>;

// CHECK-LABEL: // (sdivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs, 42:{ *:[i32] })  =>  (tst4:{ *:[i32] }:{ *:[i32] } (tst4:{ *:[i32] }:{ *:[i32] } ?:{ *:[i32] }:$rs))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(MyTarget::tst4),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/2, // rs
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/2, /*OpIdx for MyTarget::R0*/0,
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst4),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[rd]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 3,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: // 42:{ *:[i32] }  =>  (tst1:{ *:[i32] })
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst1),
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/0, /*OpIdx for MyTarget::B0*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: // (sdivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)  =>  (tst2:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst2),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[rd]
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/2, // rs1
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/3, // rs2
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/0, /*OpIdx for MyTarget::R1*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 1,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: // (udivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)  =>  (tst3:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst3),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/2, // rs1
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/3, // rs2
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // DstI[R1]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/2, GIMT_Encode2(MyTarget::R1), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 2,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
