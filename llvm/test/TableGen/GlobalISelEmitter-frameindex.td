// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common %s -o - < %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;

//===- Test a simple pattern with frame index operands. ----------------------===//
//
// CHECK:       GIM_Try, /*On fail goto*//*Label [[LABEL_NUM:[0-9]+]]*/ GIMT_Encode4([[LABEL:[0-9]+]]),
// CHECK-NEXT:    GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:    GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FRAME_INDEX),
// CHECK-NEXT:    // MIs[0] DstI[dst]
// CHECK-NEXT:    GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_p0s32,
// CHECK-NEXT:    GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT:    // MIs[0] fi
// CHECK-NEXT:    // No operand predicates
// CHECK-NEXT:    // (frameindex:{ *:[i32] }):$fi  =>  (ADD:{ *:[i32] } (tframeindex:{ *:[i32] }):$fi, 0:{ *:[i32] })
// CHECK-NEXT:    GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::ADD),
// CHECK-NEXT:    GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:    GIR_RootToRootCopy, /*OpIdx*/1, // fi
// CHECK-NEXT:    GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:    GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:    // GIR_Coverage, 0,
// CHECK-NEXT:    GIR_EraseRootFromParent_Done,
// CHECK-NEXT:  // Label [[LABEL_NUM]]: @[[LABEL]]
// CHECK-NEXT:  GIM_Reject,

def : Pat<(p0 frameindex:$fi), (ADD tframeindex:$fi, 0)>;
