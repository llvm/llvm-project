// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/Common -I %p/../../include %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let TargetPrefix = "mytarget" in {
def int_mytarget_sleep0 : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
}

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL-NEXT: // MIs[0] Operand 0
// GISEL-NEXT: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, GIMT_Encode2(Intrinsic::mytarget_sleep0),
// GISEL-NEXT: // MIs[0] src
// GISEL-NEXT: GIM_CheckIsImm, /*MI*/0, /*Op*/1,
// GISEL-NEXT: GIM_CheckImmOperandPredicate, /*MI*/0, /*MO*/1, /*Predicate*/GIMT_Encode2(GICXXPred_I64_Predicate_tuimm9),
// GISEL-NEXT: // (intrinsic_void {{[0-9]+}}:{ *:[iPTR] }, (timm:{ *:[i32] })<<P:Predicate_tuimm9>>:$src) => (SLEEP0 (timm:{ *:[i32] }):$src)
// GISEL-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::SLEEP0),
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/1, // src
def tuimm9 : TImmLeaf<i32, [{ return isUInt<9>(Imm); }]>;
def SLEEP0 : I<(outs), (ins i32imm:$src),
  [(int_mytarget_sleep0 tuimm9:$src)]
>;
