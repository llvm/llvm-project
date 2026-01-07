; RUN: opt -S -dxil-resource-implicit-binding %s | FileCheck %s

; Resources defined
; RWBuffer<float> A : register(u1);
; RWBuffer<float> B[];     // gets u6 (unbounded range)
; RWBuffer<int> C : register(u5);
; RWBuffer<float> D[3];    // gets u2 because it fits between A and C but not before A

target triple = "dxil-pc-shadermodel6.6-compute"

define void @test_unbounded_arrays() {

; RWBuffer<float> A : register(u1);
  %bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
; no change to llvm.dx.resource.handlefrombinding
; CHECK: %bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 1, i32 1, i32 0, ptr null)

; RWBuffer<float> B[];
%bufB = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 100, i32 0, i32 -1, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 6, i32 -1, i32 0, ptr null)

; RWBuffer<int> C : register(u5);
  %bufC = call target("dx.TypedBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr null)
; no change to llvm.dx.resource.handlefrombinding
; CHECK: %bufC = call target("dx.TypedBuffer", i32, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_0t(i32 0, i32 5, i32 1, i32 0, ptr null)

; ; RWBuffer<float> D[3];
  %bufD = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 101, i32 0, i32 3, i32 1, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 2, i32 3, i32 1, ptr null)

; CHECK-NOT: @llvm.dx.resource.handlefromimplicitbinding
  ret void
}

