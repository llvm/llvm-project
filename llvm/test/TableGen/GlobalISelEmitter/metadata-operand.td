// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -I %p/../../../include -I %p/../Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// No type check for a metadata operand matched via srcvalue.

def int_tgt_metadata_intrinsic : Intrinsic<
  [llvm_i32_ty],
  [llvm_ptr_ty, llvm_metadata_ty],
  []>;

def metadata_intrin_pat : PatFrag<
  (ops node:$ptr),
  (int_tgt_metadata_intrinsic $ptr, srcvalue)>;

def LOAD : I<(outs GPR32:$dst), (ins GPR32Op:$src), []>;

// CHECK: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// CHECK-NEXT: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, GIMT_Encode2(Intrinsic::tgt_metadata_intrinsic),
// CHECK-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/2, /*SizeInBits*/32,
// CHECK-NEXT: // MIs[0] Operand 3
// No type check for the metadata operand
// CHECK-NEXT: // No operand predicates
def : Pat<
  (i32 (metadata_intrin_pat p0:$src)),
  (LOAD GPR32:$src)
>;
