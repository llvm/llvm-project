// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=true -I %p/../../include -I %p/Common -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def InstTwoOperands : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def InstThreeOperands : I<(outs GPR32:$dst), (ins GPR32:$cond, GPR32:$src,GPR32:$src2), []>;
// CHECK:      GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(229),
// CHECK-NEXT:   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SELECT),
// CHECK-NEXT:   GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT:   GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:   GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:   GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(197),
// CHECK-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:     GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/3, /*OtherMI*/2, /*OtherOpIdx*/2,
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(114), // Rule ID 1 //
// CHECK-NEXT:       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ICMP),
// CHECK-NEXT:       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckType, /*MI*/1, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:       // MIs[1] Operand 1
// CHECK-NEXT:       GIM_CheckCmpPredicate, /*MI*/1, /*Op*/1, /*Predicate*/GIMT_Encode2(CmpInst::ICMP_EQ),
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:       GIM_CheckConstantInt8, /*MI*/1, /*Op*/3, 0,
// CHECK-NEXT:       GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:       GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/2, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:       // (select:{ *:[i32] } (setcc:{ *:[i32] } GPR32:{ *:[i32] }:$cond, 0:{ *:[i32] }, SETEQ:{ *:[Other] }), (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2), GPR32:{ *:[i32] }:$src2)  =>  (InstThreeOperands:{ *:[i32] } GPR32:{ *:[i32] }:$cond, GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::InstThreeOperands),
// CHECK-NEXT:       GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // cond
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src1
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // src2
// CHECK-NEXT:       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:       // GIR_Coverage, 1,
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 2: @114
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(196), // Rule ID 2 //
// CHECK-NEXT:       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ICMP),
// CHECK-NEXT:       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckType, /*MI*/1, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:       // MIs[1] Operand 1
// CHECK-NEXT:       GIM_CheckCmpPredicate, /*MI*/1, /*Op*/1, /*Predicate*/GIMT_Encode2(CmpInst::ICMP_NE),
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:       GIM_CheckConstantInt8, /*MI*/1, /*Op*/3, 0,
// CHECK-NEXT:       GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:       GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:       GIM_CheckRegBankForClass, /*MI*/2, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:       // (select:{ *:[i32] } (setcc:{ *:[i32] } GPR32:{ *:[i32] }:$cond, 0:{ *:[i32] }, SETNE:{ *:[Other] }), (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2), GPR32:{ *:[i32] }:$src2)  =>  (InstThreeOperands:{ *:[i32] } GPR32:{ *:[i32] }:$cond, GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::InstThreeOperands),
// CHECK-NEXT:       GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // cond
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src1
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // src2
// CHECK-NEXT:       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:       // GIR_Coverage, 2,
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 3: @196
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:   // Label 1: @197
// CHECK-NEXT:   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(228), // Rule ID 0 //
// CHECK-NEXT:     GIM_RootCheckType, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/3, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:     // (select:{ *:[i32] } GPR32:{ *:[i32] }:$cond, GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)  =>  (InstThreeOperands:{ *:[i32] } GPR32:{ *:[i32] }:$cond, GPR32:{ *:[i32] }:$src1, GPR32:{ *:[i32] }:$src2)
// CHECK-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::InstThreeOperands),
// CHECK-NEXT:     GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:     // GIR_Coverage, 0,
// CHECK-NEXT:     GIR_Done,
// CHECK-NEXT:   // Label 4: @228
// CHECK-NEXT:   GIM_Reject,
// CHECK-NEXT: // Label 0: @229
// CHECK-NEXT: GIM_Reject,
def : Pat<(i32 (select GPR32:$cond, GPR32:$src1, GPR32:$src2)),
          (InstThreeOperands GPR32:$cond, GPR32:$src1, GPR32:$src2)>;

def : Pat<(i32 (select (i32 (setcc GPR32:$cond, (i32 0), (OtherVT SETEQ))),
                       (i32 (sub GPR32:$src1, GPR32:$src2)),
                       GPR32:$src2)),
          (InstThreeOperands GPR32:$cond, GPR32:$src1, GPR32:$src2)>;

def : Pat<(i32 (select (i32 (setcc GPR32:$cond, (i32 0), (OtherVT SETNE))),
                       (i32 (sub GPR32:$src1, GPR32:$src2)),
                       GPR32:$src2)),
          (InstThreeOperands GPR32:$cond, GPR32:$src1, GPR32:$src2)>;
