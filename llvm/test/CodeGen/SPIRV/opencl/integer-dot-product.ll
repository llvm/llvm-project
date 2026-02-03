; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv1.6-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-16
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test OpenCL integer dot product lowering for SPIR-V.

; CHECK-SPIRV-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Int32x4Ty:]] = OpTypeVector %[[#Int32Ty]] 4

; CHECK-SPIRV-16-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0

; CHECK-SPIRV: %[[#SignedMulVec:]] = OpIMul %[[#Int32x4Ty]] %[[#SignedVec0:]] %[[#SignedVec1:]]
; CHECK-SPIRV: %[[#SignedElt0:]] = OpCompositeExtract %[[#Int32Ty]] %[[#SignedMulVec]] 0
; CHECK-SPIRV: %[[#SignedElt1:]] = OpCompositeExtract %[[#Int32Ty]] %[[#SignedMulVec]] 1
; CHECK-SPIRV: %[[#SignedSum01:]] = OpIAdd %[[#Int32Ty]] %[[#SignedElt0]] %[[#SignedElt1]]
; CHECK-SPIRV: %[[#SignedElt2:]] = OpCompositeExtract %[[#Int32Ty]] %[[#SignedMulVec]] 2
; CHECK-SPIRV: %[[#SignedSum012:]] = OpIAdd %[[#Int32Ty]] %[[#SignedSum01]] %[[#SignedElt2]]
; CHECK-SPIRV: %[[#SignedElt3:]] = OpCompositeExtract %[[#Int32Ty]] %[[#SignedMulVec]] 3
; CHECK-SPIRV: %[[#SignedRes:]] = OpIAdd %[[#Int32Ty]] %[[#SignedSum012]] %[[#SignedElt3]]
; CHECK-SPIRV: OpStore %[[#]] %[[#SignedRes]]

; CHECK-SPIRV-16: %[[#SDot:]] = OpSDot %[[#Int32Ty]] %[[#]] %[[#]]
; CHECK-SPIRV-16: OpStore %[[#]] %[[#SDot]]

define spir_kernel void @testSignedIntDot(<4 x i32> %a, <4 x i32> %b, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i32 @_Z3dotDv4_iS_(<4 x i32> %a, <4 x i32> %b)
  store i32 %call, ptr addrspace(1) %out
  ret void
}

; CHECK-SPIRV: %[[#UnsignedMulVec:]] = OpIMul %[[#Int32x4Ty]] %[[#UnsignedVec0:]] %[[#UnsignedVec1:]]
; CHECK-SPIRV: %[[#UnsignedElt0:]] = OpCompositeExtract %[[#Int32Ty]] %[[#UnsignedMulVec]] 0
; CHECK-SPIRV: %[[#UnsignedElt1:]] = OpCompositeExtract %[[#Int32Ty]] %[[#UnsignedMulVec]] 1
; CHECK-SPIRV: %[[#UnsignedSum01:]] = OpIAdd %[[#Int32Ty]] %[[#UnsignedElt0]] %[[#UnsignedElt1]]
; CHECK-SPIRV: %[[#UnsignedElt2:]] = OpCompositeExtract %[[#Int32Ty]] %[[#UnsignedMulVec]] 2
; CHECK-SPIRV: %[[#UnsignedSum012:]] = OpIAdd %[[#Int32Ty]] %[[#UnsignedSum01]] %[[#UnsignedElt2]]
; CHECK-SPIRV: %[[#UnsignedElt3:]] = OpCompositeExtract %[[#Int32Ty]] %[[#UnsignedMulVec]] 3
; CHECK-SPIRV: %[[#UnsignedRes:]] = OpIAdd %[[#Int32Ty]] %[[#UnsignedSum012]] %[[#UnsignedElt3]]
; CHECK-SPIRV: OpStore %[[#]] %[[#UnsignedRes]]

; CHECK-SPIRV-16: %[[#UDot:]] = OpUDot %[[#Int32Ty]] %[[#]] %[[#]]
; CHECK-SPIRV-16: OpStore %[[#]] %[[#UDot]]

define spir_kernel void @testUnsignedIntDot(<4 x i32> %a, <4 x i32> %b, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i32 @_Z3dotDv4_jS_(<4 x i32> %a, <4 x i32> %b)
  store i32 %call, ptr addrspace(1) %out
  ret void
}

declare spir_func i32 @_Z3dotDv4_iS_(<4 x i32>, <4 x i32>)

declare spir_func i32 @_Z3dotDv4_jS_(<4 x i32>, <4 x i32>)
