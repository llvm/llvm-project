// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s -o - | FileCheck %s

// Boilerplate code.
include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let TargetPrefix = "mytarget" in {
    def int_mytarget_anyptr : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty]>;
}

// Ensure that llvm_anyptr_ty on an intrinsic results in a
// GIM_CheckPointerToAny rather than a GIM_CheckType.
//
// CHECK: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::mytarget_anyptr,
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/2, /*SizeInBits*/32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/MyTarget::GPR32RegClassID,
// CHECK-NEXT: GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GICXXPred_MI_Predicate_frag_anyptr,
// CHECK-NEXT: // (intrinsic_w_chain:{ *:[i32] } {{[0-9]+}}:{ *:[iPTR] }, GPR32:{ *:[i32] }:$src)<<P:Predicate_frag_anyptr>>  =>  (ANYLOAD:{ *:[i32] } GPR32:{ *:[i32] }:$src)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::ANYLOAD,
let hasSideEffects = 1 in {
  def ANYLOAD : I<(outs GPR32:$dst), (ins GPR32:$src1),
            [(set GPR32:$dst, (load GPR32:$src1))]>;
}

def frag_anyptr : PatFrag<(ops node:$src),
                   (int_mytarget_anyptr node:$src),
                   [{ return true; // C++ code }]> {
  let GISelPredicateCode = [{ return true; // C++ code }];
}

def : Pat<(frag_anyptr GPR32:$src),
          (p0 (ANYLOAD GPR32:$src))>;
