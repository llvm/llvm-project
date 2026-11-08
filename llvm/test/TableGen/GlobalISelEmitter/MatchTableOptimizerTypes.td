// RUN: llvm-tblgen %s -gen-global-isel -gisel-extended-llt -optimize-match-table=true -I %p/../../../include -I %p/../Common | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Check the MatchTable does not remove necessary type checks

def V1 : Register<"v1"> { let Namespace = "MyTarget"; }
def FPR64  : RegisterClass<"MyTarget", [v4f16, v4bf16], 64, (add V1)>;
def FPR128 : RegisterClass<"MyTarget", [v4f32, v8f16, v8bf16], 128, (add V1)>;
def BF16I  : I<(outs FPR128:$dst), (ins FPR64:$Rn, FPR128:$Rm), []>;
def F16I   : I<(outs FPR128:$dst), (ins FPR64:$Rn, FPR128:$Rm), []>;

def : Pat<(v8bf16 (concat_vectors (v4bf16 FPR64:$Rn), (v4bf16 (fpround (v4f32 FPR128:$Rm))))),
          (BF16I $Rn, $Rm)>;
def : Pat<(v8f16 (concat_vectors (v4f16 FPR64:$Rn), (v4f16 (fpround (v4f32 FPR128:$Rm))))),
          (F16I $Rn, $Rm)>;

// CHECK:      constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:   /*   0 */ GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(112),
// CHECK-NEXT:   /*   5 */   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_CONCAT_VECTORS),
// CHECK-NEXT:   /*   9 */   GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:   /*  12 */   GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_v8s16,
// CHECK-NEXT:   /*  15 */   GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_v4s16,
// CHECK-NEXT:   /*  18 */   GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_v4s16,
// CHECK-NEXT:   /*  21 */   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::FPR128RegClassID),
// CHECK-NEXT:   /*  25 */   GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::FPR64RegClassID),
// CHECK-NEXT:   /*  29 */   GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(70), // Rule ID 0 //
// CHECK-NEXT:   /*  34 */     GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:   /*  38 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FPTRUNC),
// CHECK-NEXT:   /*  42 */     GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_v4bf16,
// CHECK-NEXT:   /*  46 */     GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4f32,
// CHECK-NEXT:   /*  50 */     GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::FPR128RegClassID),
// CHECK-NEXT:   /*  55 */     GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:   /*  57 */     // (concat_vectors:{ *:[v8bf16] } FPR64:{ *:[v4bf16] }:$Rn, (fpround:{ *:[v4bf16] } FPR128:{ *:[v4f32] }:$Rm))  =>  (BF16I:{ *:[v8bf16] } ?:{ *:[v4bf16] }:$Rn, ?:{ *:[v4f32] }:$Rm)
// CHECK-NEXT:   /*  57 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::BF16I),
// CHECK-NEXT:   /*  60 */     GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:   /*  62 */     GIR_RootToRootCopy, /*OpIdx*/1, // Rn
// CHECK-NEXT:   /*  64 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // Rm
// CHECK-NEXT:   /*  68 */     GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:   /*  69 */     // GIR_Coverage, 0,
// CHECK-NEXT:   /*  69 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:   /*  70 */   // Label 1: @70
// CHECK-NEXT:   /*  70 */   GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(111), // Rule ID 1 //
// CHECK-NEXT:   /*  75 */     GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:   /*  79 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FPTRUNC),
// CHECK-NEXT:   /*  83 */     GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_v4f16,
// CHECK-NEXT:   /*  87 */     GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4f32,
// CHECK-NEXT:   /*  91 */     GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::FPR128RegClassID),
// CHECK-NEXT:   /*  96 */     GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:   /*  98 */     // (concat_vectors:{ *:[v8f16] } FPR64:{ *:[v4f16] }:$Rn, (fpround:{ *:[v4f16] } FPR128:{ *:[v4f32] }:$Rm))  =>  (F16I:{ *:[v8f16] } ?:{ *:[v4f16] }:$Rn, ?:{ *:[v4f32] }:$Rm)
// CHECK-NEXT:   /*  98 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::F16I),
// CHECK-NEXT:   /* 101 */     GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:   /* 103 */     GIR_RootToRootCopy, /*OpIdx*/1, // Rn
// CHECK-NEXT:   /* 105 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // Rm
// CHECK-NEXT:   /* 109 */     GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:   /* 110 */     // GIR_Coverage, 1,
// CHECK-NEXT:   /* 110 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:   /* 111 */   // Label 2: @111
// CHECK-NEXT:   /* 111 */   GIM_Reject,
// CHECK-NEXT:   /* 112 */ // Label 0: @112
// CHECK-NEXT:   /* 112 */ GIM_Reject,
// CHECK-NEXT:   /* 113 */ }; // Size: 113 bytes
