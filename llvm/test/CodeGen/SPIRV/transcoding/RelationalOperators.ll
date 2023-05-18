; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:      %[[#bool:]] = OpTypeBool
; CHECK-SPIRV:      %[[#bool2:]] = OpTypeVector %[[#bool]] 2

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpUGreaterThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testUGreaterThan(uint2 a, uint2 b, global int2 *res) {
;;   res[0] = a > b;
;; }

define dso_local spir_kernel void @testUGreaterThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp ugt <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpSGreaterThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testSGreaterThan(int2 a, int2 b, global int2 *res) {
;;   res[0] = a > b;
;; }

define dso_local spir_kernel void @testSGreaterThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp sgt <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpUGreaterThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testUGreaterThanEqual(uint2 a, uint2 b, global int2 *res) {
;;   res[0] = a >= b;
;; }

define dso_local spir_kernel void @testUGreaterThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp uge <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpSGreaterThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testSGreaterThanEqual(int2 a, int2 b, global int2 *res) {
;;   res[0] = a >= b;
;; }

define dso_local spir_kernel void @testSGreaterThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp sge <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpULessThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testULessThan(uint2 a, uint2 b, global int2 *res) {
;;   res[0] = a < b;
;; }

define dso_local spir_kernel void @testULessThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp ult <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpSLessThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testSLessThan(int2 a, int2 b, global int2 *res) {
;;   res[0] = a < b;
;; }

define dso_local spir_kernel void @testSLessThan(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp slt <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpULessThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testULessThanEqual(uint2 a, uint2 b, global int2 *res) {
;;   res[0] = a <= b;
;; }

define dso_local spir_kernel void @testULessThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp ule <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpSLessThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testSLessThanEqual(int2 a, int2 b, global int2 *res) {
;;   res[0] = a <= b;
;; }

define dso_local spir_kernel void @testSLessThanEqual(<2 x i32> noundef %a, <2 x i32> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = icmp sle <2 x i32> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFOrdEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFOrdEqual(float2 a, float2 b, global int2 *res) {
;;   res[0] = a == b;
;; }

define dso_local spir_kernel void @testFOrdEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = fcmp oeq <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFUnordNotEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFUnordNotEqual(float2 a, float2 b, global int2 *res) {
;;   res[0] = a != b;
;; }

define dso_local spir_kernel void @testFUnordNotEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = fcmp une <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFOrdGreaterThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFOrdGreaterThan(float2 a, float2 b, global int2 *res) {
;;   res[0] = a > b;
;; }

define dso_local spir_kernel void @testFOrdGreaterThan(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = fcmp ogt <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFOrdGreaterThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFOrdGreaterThanEqual(float2 a, float2 b, global int2 *res) {
;;   res[0] = a >= b;
;; }

define dso_local spir_kernel void @testFOrdGreaterThanEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = fcmp oge <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFOrdLessThan %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFOrdLessThan(float2 a, float2 b, global int2 *res) {
;;   res[0] = a < b;
;; }

define dso_local spir_kernel void @testFOrdLessThan(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = fcmp olt <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV-NEXT: %[[#A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV:      %[[#]] = OpFOrdLessThanEqual %[[#bool2]] %[[#A]] %[[#B]]
; CHECK-SPIRV:      OpFunctionEnd

;; kernel void testFOrdLessThanEqual(float2 a, float2 b, global int2 *res) {
;;   res[0] = a <= b;
;; }

define dso_local spir_kernel void @testFOrdLessThanEqual(<2 x float> noundef %a, <2 x float> noundef %b, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %cmp = fcmp ole <2 x float> %a, %b
  %sext = sext <2 x i1> %cmp to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %res, align 8
  ret void
}
