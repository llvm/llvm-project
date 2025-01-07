// RUN: llvm-tblgen -gen-global-isel -I %p/../../../include -I %p/../Common %s | FileCheck %s

// Boilerplate code.
include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def GPR : RegisterClass<"MyTarget", [i32, i64], 32, (add R0)>;

let TargetPrefix = "mytarget" in {
    def int_mytarget_anyptr : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty]>;
}

// Check that iPTR in the destination DAG doesn't prevent the pattern from being imported.

// CHECK: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src1
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/0,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT: // (ld:{ *:[i32] } GPR:{ *:[iPTR] }:$src1)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (ANYLOAD:{ *:[i32] } GPR:{ *:[iPTR] }:$src1)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::ANYLOAD),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_Done,

let hasSideEffects = 1 in {
  def ANYLOAD : I<(outs GPR32:$dst), (ins GPR:$src1),
                  [(set GPR32:$dst, (load GPR:$src1))]>;
}

// Ensure that llvm_anyptr_ty on an intrinsic results in a
// GIM_CheckPointerToAny rather than a GIM_CheckType.
//
// CHECK: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, GIMT_Encode2(Intrinsic::mytarget_anyptr),
// CHECK-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/2, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_frag_anyptr),
// CHECK-NEXT: // (intrinsic_w_chain:{ *:[i32] } {{[0-9]+}}:{ *:[iPTR] }, GPR32:{ *:[i32] }:$src)<<P:Predicate_frag_anyptr>>  =>  (ANYLOAD:{ *:[i32] } GPR32:{ *:[i32] }:$src)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::ANYLOAD),

def frag_anyptr : PatFrag<(ops node:$src),
                   (int_mytarget_anyptr node:$src),
                   [{ return true; // C++ code }]> {
  let GISelPredicateCode = [{ return true; // C++ code }];
}

def : Pat<(frag_anyptr GPR32:$src),
          (p0 (ANYLOAD GPR32:$src))>;
