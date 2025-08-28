; RUN: opt -S -dxil-resource-implicit-binding %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @test_simple_binding() {

; StructuredBuffer<float> A : register(t1);
  %bufA = call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
; no change to llvm.dx.resource.handlefrombinding
; CHECK: %bufA = call target("dx.RawBuffer", float, 0, 0) 
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 1, i32 1, i32 0, ptr null)

; StructuredBuffer<float> B; // gets register(t0, space0)
  %bufB = call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 5, i32 0, i32 1, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.RawBuffer", float, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 0, i32 1, i32 0, ptr null)

; StructuredBuffer<float> C; // gets register(t2, space0)
  %bufC = call target("dx.RawBuffer", float, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 6, i32 0, i32 1, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.RawBuffer", float, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_0_0t(i32 0, i32 2, i32 1, i32 0, ptr null)

; CHECK-NOT: @llvm.dx.resource.handlefromimplicitbinding

  ret void
}

