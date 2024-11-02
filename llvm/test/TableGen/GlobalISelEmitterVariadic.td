// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def : GINodeEquiv<G_BUILD_VECTOR, build_vector>;

def ONE : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def TWO : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;

// G_BUILD_VECTOR is guaranteed to have at least one operand, therefore performing a
// number of operands check is not needed to avoid per-operand checks accessing the
// MI's operand list out of bounds in this particular pattern. However, we still
// must perform the check, as a G_BUILD_VECTOR instance with two source operands
// will pass all checks done by this pattern otherwise, which will lead to a
// mis-match if this pattern tried first (and it will if it has higher complexity).
def : Pat<(build_vector GPR32:$src1),
          (ONE GPR32:$src1)> {
  let AddedComplexity = 1000;
}

def : Pat<(build_vector GPR32:$src1, GPR32:$src2),
          (TWO GPR32:$src1, GPR32:$src2)>;

// CHECK:       GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4([[NEXT_OPCODE_LABEL:[0-9]+]]),
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR),
// CHECK-NEXT:    GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4([[NEXT_NUM_OPERANDS_LABEL_1:[0-9]+]]), // Rule ID 0 //
// CHECK-NEXT:      GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:      GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:      // (build_vector:{ *:[i32] } GPR32:{ *:[i32] }:$src1)  =>  (ONE:{ *:[i32] } GPR32:{ *:[i32] }:$src1)
// CHECK-NEXT:      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::ONE),
// CHECK-NEXT:      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:      // GIR_Coverage, 0,
// CHECK-NEXT:      GIR_Done,
// CHECK-NEXT:    // Label 1: @[[NEXT_NUM_OPERANDS_LABEL_1]]
// CHECK-NEXT:    GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4([[NEXT_NUM_OPERANDS_LABEL_2:[0-9]+]]), // Rule ID 1 //
// CHECK-NEXT:      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:      GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:      // (build_vector:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)  =>  (TWO:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)
// CHECK-NEXT:      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::TWO),
// CHECK-NEXT:      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT:      // GIR_Coverage, 1,
// CHECK-NEXT:      GIR_Done,
// CHECK-NEXT:    // Label 2: @[[NEXT_NUM_OPERANDS_LABEL_2]]
// CHECK-NEXT:    GIM_Reject,
// CHECK-NEXT:  // Label 0: @[[NEXT_OPCODE_LABEL]]
// CHECK-NEXT:  GIM_Reject,
