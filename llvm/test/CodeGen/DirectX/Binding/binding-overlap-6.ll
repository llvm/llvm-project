; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; Check overlap with unbounded array

; A overlaps with B
; RWBuffer<float> A[3] : register(u0);
; RWBuffer<float> B[] : register(u4);
; RWBuffer<float> C : register(u17);

; CHECK: error: resource B at register 4 overlaps with resource C at register 17 in space 0

target triple = "dxil-pc-shadermodel6.3-library"

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
@C.str = private unnamed_addr constant [2 x i8] c"C\00", align 1

define void @test_overlapping() {
entry:
  %h1 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 3, i32 0, ptr @A.str)
  %h2 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 4, i32 -1, i32 0, ptr @B.str)
  %h3 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 17, i32 1, i32 0, ptr @C.str)
  ret void
}
