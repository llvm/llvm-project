; RUN: opt -S -dxil-resource-implicit-binding %s | FileCheck %s

; Resources defined (with random order of handlefromimplicitbinding calls):
; RWBuffer<float> A : register(u2);
; RWBuffer<float> B[4];    // gets u3 because it does not fit before A (range 4)
; RWBuffer<int> C[2];      // gets u0 because it fits before A (range 2)
; RWBuffer<float> E[5];    // gets u7 which is right after B (range 5)

target triple = "dxil-pc-shadermodel6.6-compute"

define void @test_arrays() {

; RWBuffer<float> A : register(u2);
  %bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
; no change to llvm.dx.resource.handlefrombinding
; CHECK: %bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 2, i32 1, i32 0, ptr null)

; RWBuffer<float> E[2];
  %bufE = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 30, i32 0, i32 5, i32 4, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 7, i32 5, i32 4, ptr null)

; RWBuffer<float> B[4];
  %bufB = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 10, i32 0, i32 4, i32 2, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 3, i32 4, i32 2, ptr null)

; RWBuffer<int> C[2];
  %bufC = call target("dx.TypedBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 20, i32 0, i32 2, i32 1, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", i32, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_0t(i32 0, i32 0, i32 2, i32 1, ptr null)

; another access to resource array B to make sure it gets the same binding
  %bufB2 = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 10, i32 0, i32 4, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 3, i32 4, i32 0, ptr null)

; CHECK-NOT: @llvm.dx.resource.handlefromimplicitbinding
  ret void
}

