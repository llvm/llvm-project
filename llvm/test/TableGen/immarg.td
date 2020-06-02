// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/Common -I %p/../../include %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let TargetPrefix = "mytarget" in {
def int_mytarget_sleep0 : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_mytarget_sleep1 : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
}

// GISEL: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS,
// GISEL-NEXT: // MIs[0] Operand 0
// GISEL-NEXT: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::mytarget_sleep0,
// GISEL-NEXT: // MIs[0] src
// GISEL-NEXT: GIM_CheckIsImm, /*MI*/0, /*Op*/1,
// GISEL-NEXT: // (intrinsic_void {{[0-9]+}}:{ *:[iPTR] }, (timm:{ *:[i32] }):$src)  =>  (SLEEP0 (timm:{ *:[i32] }):$src)
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::SLEEP0,
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
def SLEEP0 : I<(outs), (ins i32imm:$src),
  [(int_mytarget_sleep0 timm:$src)]
>;

// Test for situation which was crashing in ARM patterns.
def p_imm : Operand<i32>;
def SLEEP1 : I<(outs), (ins p_imm:$src), []>;

// FIXME: This should not crash, but should it work or be an error?
// def : Pat <
//   (int_mytarget_sleep1 timm:$src),
//   (SLEEP1 imm:$src)
// >;
