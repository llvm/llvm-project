// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common %s -o - < %s | FileCheck -check-prefix=SDAG %s
// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Test the HasNoUse predicate

def NO_RET_ATOMIC_ADD : I<(outs), (ins GPR32Op:$src0, GPR32Op:$src1), []>;

// SDAG: case 0: {
// SDAG-NEXT: // Predicate_atomic_load_add_no_ret_32
// SDAG-NEXT: SDNode *N = Node;
// SDAG-NEXT: (void)N;
// SDAG-NEXT: if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
// SDAG-NEXT: if (!SDValue(N, 0).use_empty()) return false;
// SDAG-NEXT: return true;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ATOMICRMW_ADD),
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/GIMT_Encode4(4),
// GISEL-NEXT: GIM_CheckHasNoUse, /*MI*/0,
// GISEL-NEXT: // MIs[0] src0
// GISEL-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/0,
// GISEL-NEXT: // (atomic_load_add:{ *:[i32] } iPTR:{ *:[iPTR] }:$src0, i32:{ *:[i32] }:$src1)<<P:Predicate_atomic_load_add_no_ret_32>>  =>  (NO_RET_ATOMIC_ADD GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::NO_RET_ATOMIC_ADD),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
// GISEL-NEXT: GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/1, /*MergeInsnID's*/0,
// GISEL-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// GISEL-NEXT: GIR_EraseFromParent, /*InsnID*/0,
let HasNoUse = true in
defm atomic_load_add_no_ret : binary_atomic_op<atomic_load_add>;

def : Pat <
  (atomic_load_add_no_ret_32 iPTR:$src0, i32:$src1),
  (NO_RET_ATOMIC_ADD GPR32:$src0, GPR32:$src1)
>;
