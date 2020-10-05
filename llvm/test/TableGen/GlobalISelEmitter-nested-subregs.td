// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let Namespace = "MyTarget" in {

def lo8  : SubRegIndex<8>;
def hi8  : SubRegIndex<8, 8>;
def lo16 : SubRegIndex<16>;
def hi16 : SubRegIndex<16, 16>;

def a0bl : Register<"a0bl">;
def a0bh : Register<"a0bh">;
def a0wh : Register<"a0wh">;

} // Namespace = "MyTarget"

def a0wl: RegisterWithSubRegs<"a0", [a0bh, a0bl]> {
  let SubRegIndices = [hi8, lo8];
  let CoveredBySubRegs = 1;
}

def a0: RegisterWithSubRegs<"a0", [a0wh, a0wl]> {
  let SubRegIndices = [hi16, lo16];
  let CoveredBySubRegs = 1;
}

def A0b : RegisterClass<"MyTarget",  [i8],  8, (add a0bl)>;
def A0w : RegisterClass<"MyTarget", [i16], 16, (add a0wl)>;
def A0  : RegisterClass<"MyTarget", [i32], 32, (add a0)>;

// CHECK: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ANYEXT,
// CHECK-NEXT: // MIs[0] dst
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s16,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::A0RegClassID,
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
// CHECK-NEXT: // (anyext:{ *:[i16] } i8:{ *:[i8] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i16] } (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), A0b:{ *:[i8] }:$src, lo8:{ *:[i32] }), lo16:{ *:[i32] })
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::INSERT_SUBREG,
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_AddImm, /*InsnID*/1, /*Imm*/3,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, MyTarget::A0RegClassID,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, MyTarget::A0RegClassID,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/2, MyTarget::A0bRegClassID,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT: GIR_AddTempSubRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0, MyTarget::lo16,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, MyTarget::A0wRegClassID,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, MyTarget::A0RegClassID,
def : Pat<(i16 (anyext i8:$src)),
          (i16 (EXTRACT_SUBREG
                 (i32 (INSERT_SUBREG
                        (i32 (IMPLICIT_DEF)),
                        A0b:$src,
                        lo8)),
                 lo16))>;
