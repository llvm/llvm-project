; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; Requirements derived from an operand type must be found when the operand is a PHI.

; CHECK-DAG: OpCapability Int64Atomics
; CHECK-DAG: OpCapability DotProduct
; CHECK-DAG: OpCapability DotProductInput4x8BitPacked
; CHECK-DAG: OpExtension "SPV_KHR_integer_dot_product"
; CHECK: OpMemoryModel
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0

; CHECK-LABEL: Begin function fcmp_phi
; CHECK: %[[#FX:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#FY:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#FV:]] = OpPhi %[[#Float]] %[[#FX]] %[[#]] %[[#FY]] %[[#]]
; CHECK: OpFOrdLessThan %[[#Bool]] %[[#FV]] %[[#FY]]
define spir_func i1 @fcmp_phi(float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit
then:
  br label %exit
exit:
  %v = phi float [ %x, %entry ], [ %y, %then ]
  %r = fcmp olt float %v, %y
  ret i1 %r
}

; CHECK-LABEL: Begin function atomic_store_phi
; CHECK: %[[#AX:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#AY:]] = OpFunctionParameter %[[#I64]]
; CHECK: OpFunctionParameter %[[#Bool]]
; CHECK: %[[#AP:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#AV:]] = OpPhi %[[#I64]] %[[#AX]] %[[#]] %[[#AY]] %[[#]]
; CHECK: OpAtomicStore %[[#AP]] %[[#]] %[[#]] %[[#AV]]
define spir_func void @atomic_store_phi(i64 %x, i64 %y, i1 %c, ptr addrspace(1) %p) {
entry:
  br i1 %c, label %then, label %exit
then:
  br label %exit
exit:
  %v = phi i64 [ %x, %entry ], [ %y, %then ]
  store atomic i64 %v, ptr addrspace(1) %p seq_cst, align 8
  ret void
}

; CHECK-LABEL: Begin function dot_phi
; CHECK: %[[#DAcc:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#DX:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#DY:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#DV:]] = OpPhi %[[#I32]] %[[#DX]] %[[#]] %[[#DY]] %[[#]]
; CHECK: %[[#Dot:]] = OpSDot %[[#I32]] %[[#DV]] %[[#DY]] PackedVectorFormat4x8Bit
; CHECK: OpIAdd %[[#I32]] %[[#Dot]] %[[#DAcc]]
define spir_func i32 @dot_phi(i32 %acc, i32 %x, i32 %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit
then:
  br label %exit
exit:
  %v = phi i32 [ %x, %entry ], [ %y, %then ]
  %r = call i32 @llvm.spv.dot4add.i8packed(i32 %acc, i32 %v, i32 %y)
  ret i32 %r
}
