; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; Check overlap error for two resource arrays.

; A overlaps with B
; RWBuffer<float> A[10] : register(u0);
; RWBuffer<float> B[10] : register(u5);

; CHECK: error: resource A at register 0 overlaps with resource B at register 5 in space 0

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1

define void @test_overlapping() {
entry:
  %h1 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 4, ptr @A.str)
  %h2 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 10, i32 4, ptr @B.str)
  ret void
}
