; RUN: not opt -S -dxil-resource-implicit-binding %s  2>&1 | FileCheck %s

; Resources defined
; RWBuffer<float> A : register(u1);
; RWBuffer<float> B[];     // gets u6 (unbounded range)
; RWBuffer<float> C : register(u5);
; RWBuffer<float> D[4];    // error - does not fit in the remaining descriptor ranges in space0 

; CHECK: error:
; CHECK-SAME: resource cannot be allocated

target triple = "dxil-pc-shadermodel6.6-compute"

define void @test_many_spaces() {

; RWBuffer<float> A : register(u1);
%bufA = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)

; RWBuffer<float> B[];
%bufB = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 100, i32 0, i32 -1, i32 0, ptr null)

; RWBuffer<int> C : register(u5);
%bufC = call target("dx.TypedBuffer", i32, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr null)

; RWBuffer<float> D[4];
%bufD = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefromimplicitbinding(i32 101, i32 0, i32 4, i32 1, ptr null)

  ret void
}

