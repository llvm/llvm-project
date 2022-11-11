; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#int2:]] = OpTypeVector %[[#int]] 2
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#float2:]] = OpTypeVector %[[#float]] 2

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpSDiv %[[#int2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testSDiv(int2 a, int2 b, global int2 *res) {
;;   res[0] = a / b;
;; }

define dso_local spir_kernel void @testSDiv(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %div = sdiv <2 x i32> %a, %b
  store <2 x i32> %div, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpUDiv %[[#int2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testUDiv(uint2 a, uint2 b, global uint2 *res) {
;;   res[0] = a / b;
;; }

define dso_local spir_kernel void @testUDiv(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %div = udiv <2 x i32> %a, %b
  store <2 x i32> %div, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFDiv %[[#float2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFDiv(float2 a, float2 b, global float2 *res) {
;;   res[0] = a / b;
;; }

define dso_local spir_kernel void @testFDiv(<2 x float> noundef %a, <2 x float> noundef %b, <2 x float> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %div = fdiv <2 x float> %a, %b
  store <2 x float> %div, <2 x float> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpSRem %[[#int2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testSRem(int2 a, int2 b, global int2 *res) {
;;   res[0] = a % b;
;; }

define dso_local spir_kernel void @testSRem(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %rem = srem <2 x i32> %a, %b
  store <2 x i32> %rem, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpUMod %[[#int2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testUMod(uint2 a, uint2 b, global uint2 *res) {
;;   res[0] = a % b;
;; }

define dso_local spir_kernel void @testUMod(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %rem = urem <2 x i32> %a, %b
  store <2 x i32> %rem, <2 x i32> addrspace(1)* %res, align 8
  ret void
}
