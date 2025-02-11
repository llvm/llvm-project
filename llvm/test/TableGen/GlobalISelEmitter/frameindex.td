// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../../include -I %p/../Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def ADDI : I<(outs GPR32:$dst), (ins GPR32:$src1, i32imm:$src2), []>;

def to_tframeindex : SDNodeXForm<frameindex, [{}]>;

def : GICustomOperandRenderer<"renderFrameIndex">,
      GISDNodeXFormEquiv<to_tframeindex>;

def : Pat<(frameindex:$fi), (ADDI (to_tframeindex $fi), 0)>;

def : Pat<(ptradd frameindex:$fi, (i32 imm:$offset)),
          (ADDI (to_tframeindex $fi), imm:$offset)>;

// CHECK-LABEL: GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(74), // Rule ID 1 //
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_PTR_ADD),
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/0, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] fi
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/1, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FRAME_INDEX),
// CHECK-NEXT: // MIs[1] Operand 0
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/1, /*Op*/0, /*SizeInBits*/32,
// CHECK-NEXT: // MIs[1] Operand 1
// CHECK-NEXT: // No operand predicates
// CHECK-NEXT: // MIs[0] offset
// CHECK-NEXT: GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/2, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_CONSTANT),
// CHECK-NEXT: // MIs[2] Operand 0
// CHECK-NEXT: GIM_CheckType, /*MI*/2, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: // MIs[2] Operand 1
// CHECK-NEXT: // No operand predicates
// CHECK-NEXT: GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT: // (ptradd:{ *:[i32] } (frameindex:{ *:[i32] }):$fi, (imm:{ *:[i32] }):$offset)  =>  (ADDI:{ *:[i32] } (to_tframeindex:{ *:[i32] } ?:{ *:[i32] }:$fi), (imm:{ *:[i32] }):$offset)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::ADDI),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_CustomRenderer, /*InsnID*/0, /*OldInsnID*/1, /*Renderer*/GIMT_Encode2(GICR_renderFrameIndex), // fi
// CHECK-NEXT: GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/2, // offset
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 1,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(109), // Rule ID 0 //
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FRAME_INDEX),
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/0, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] Operand 1
// CHECK-NEXT: // No operand predicates
// CHECK-NEXT: // (frameindex:{ *:[i32] }):$fi  =>  (ADDI:{ *:[i32] } (to_tframeindex:{ *:[i32] } ?:{ *:[i32] }:$fi), 0:{ *:[i32] })
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::ADDI),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_CustomRenderer, /*InsnID*/0, /*OldInsnID*/0, /*Renderer*/GIMT_Encode2(GICR_renderFrameIndex), // fi
// CHECK-NEXT: GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
