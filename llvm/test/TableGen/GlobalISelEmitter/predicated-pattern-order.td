// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../../include -I %p/../Common %s | FileCheck %s
// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=true -I %p/../../../include -I %p/../Common %s | FileCheck -check-prefix=OPT %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Check that IPM_GenericPredicate doesn't influence the final order of patterns.
// https://github.com/llvm/llvm-project/issues/121446

def aligned_store: PatFrag<(ops node:$v, node:$a), (store $v, $a), [{
  return true;
}]>{
  let GISelPredicateCode = [{ return true; }];
}

// CHECK: GIM_Try
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// CHECK-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT: // MIs[0] src0
// CHECK-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src1
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_aligned_store),
// CHECK-NEXT: // (st GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedstore>><<P:Predicate_store>><<P:Predicate_aligned_store>>  =>  (MOVALIGNED GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MOVALIGNED),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage

// CHECK: GIM_Try
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// CHECK-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT: // MIs[0] src0
// CHECK-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src1
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (st GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MOVUNALIGNED GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MOVUNALIGNED),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage

// OPT: GIM_Try
// OPT-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// OPT-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// OPT-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// OPT-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,

// OPT-NEXT: GIM_Try
// OPT-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// OPT-NEXT:   // MIs[0] src1
// OPT-NEXT:   GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// OPT-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// OPT-NEXT:   GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_aligned_store),
// OPT-NEXT:   // (st GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedstore>><<P:Predicate_store>><<P:Predicate_aligned_store>>  =>  (MOVALIGNED GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// OPT-NEXT:   GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MOVALIGNED),
// OPT-NEXT:   GIR_RootConstrainSelectedInstOperands,
// OPT-NEXT:   // GIR_Coverage

// OPT:      GIM_Try
// OPT-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// OPT-NEXT:   // MIs[0] src1
// OPT-NEXT:   GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// OPT-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// OPT-NEXT:   // (st GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MOVUNALIGNED GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// OPT-NEXT:   GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MOVUNALIGNED),
// OPT-NEXT:   GIR_RootConstrainSelectedInstOperands,
// OPT-NEXT:   // GIR_Coverage

def MOVALIGNED : I<(outs), (ins GPR32:$src0, GPR32:$src1),
    [(aligned_store GPR32:$src0, GPR32:$src1)]>;


def MOVUNALIGNED : I<(outs), (ins GPR32:$src0, GPR32:$src1),
    [(store GPR32:$src0, GPR32:$src1)]>;

