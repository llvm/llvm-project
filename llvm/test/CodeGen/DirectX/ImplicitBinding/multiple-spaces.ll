; RUN: opt -S -dxil-resource-implicit-binding %s | FileCheck %s

; Resources defined (with random order of handlefromimplicitbinding calls):
; RWBuffer<float> A : register(u5);           // defaults to space0
; RWBuffer<int> B[];                          // gets u6 (unbounded range)
; RWBuffer<float> C[4] : registcer(space5);    // gets u0 in space5
; RWBuffer<int> D[] : register(space5);       // gets u4 in space5
; RWBuffer<float> E[3] : register(space10);   // gets u0, space10
; StructuredBuffer<int> F : register(space3); // gets t0 in space3

target triple = "dxil-pc-shadermodel6.6-compute"

define void @test_many_spaces() {

; RWBuffer<float> A : register(u5);
  %bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr null)
; no change to llvm.dx.resource.handlefrombinding
; CHECK: %bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 5, i32 1, i32 0, ptr null)

; RWBuffer<int> B[];
%bufB = call target("dx.TypedBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 100, i32 0, i32 -1, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", i32, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_0t(i32 0, i32 6, i32 -1, i32 0, ptr null)

; RWBuffer<float> C[4] : register(space5);
%bufC = call target("dx.TypedBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 101, i32 5, i32 4, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", i32, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_0t(i32 5, i32 0, i32 4, i32 0, ptr null)

; RWBuffer<int> D[] : register(space5);
  %bufD = call target("dx.TypedBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 102, i32 5, i32 -1, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", i32, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0_0t(i32 5, i32 4, i32 -1, i32 0, ptr null)

; RWBuffer<float> E[3] : register(space10); // gets u0, space10
%bufE = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 103, i32 10, i32 4, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.TypedBuffer", float, 1, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 10, i32 0, i32 4, i32 0, ptr null)

; StructuredBuffer<int> F : register(space3); // gets t0 in space3
%bufF = call target("dx.RawBuffer", i32, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 104, i32 3, i32 1, i32 0, ptr null)
; CHECK: %{{.*}} = call target("dx.RawBuffer", i32, 0, 0)
; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 3, i32 0, i32 1, i32 0, ptr null)

; CHECK-NOT: @llvm.dx.resource.handlefromimplicitbinding
  ret void
}
