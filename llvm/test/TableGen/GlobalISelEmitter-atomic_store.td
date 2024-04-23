// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def ST_ATOM_B32 : I<(outs), (ins GPR32Op:$val, GPR32Op:$ptr), []>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// GISEL-NEXT: GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/GIMT_Encode4(1),
// GISEL-NEXT: GIM_CheckAtomicOrderingOrStrongerThan, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::Unordered,
// GISEL-NEXT: // MIs[0] val
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: // MIs[0] ptr
// GISEL-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/0,
// GISEL-NEXT: // (atomic_store i32:{ *:[i32] }:$val, iPTR:{ *:[iPTR] }:$ptr)<<P:Predicate_atomic_store_8>>  =>  (ST_ATOM_B32 GPR32Op:{ *:[i32] }:$val, GPR32Op:{ *:[i32] }:$ptr)
// GISEL-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::ST_ATOM_B32),
def : Pat<
//  (atomic_store_8 iPTR:$ptr, i32:$val),
  (atomic_store_8 i32:$val, iPTR:$ptr),
  (ST_ATOM_B32 GPR32Op:$val, GPR32Op:$ptr)
>;
