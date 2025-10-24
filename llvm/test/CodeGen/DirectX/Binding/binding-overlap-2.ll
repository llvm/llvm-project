; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; Check overlap error for two resources with identical binding

; R overlaps exactly with S
; RWBuffer<float> R : register(u5, space10);
; RWBuffer<float> S : register(u5, space10);

; CHECK: error: resource R at register 5 overlaps with resource S at register 5 in space 10

@R.str = private unnamed_addr constant [2 x i8] c"R\00", align 1
@S.str = private unnamed_addr constant [2 x i8] c"S\00", align 1

define void @test_overlapping() {
entry:
  %h1 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 10, i32 5, i32 1, i32 0, ptr @R.str)
  %h2 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 10, i32 5, i32 1, i32 0, ptr @S.str)
  ret void
}
