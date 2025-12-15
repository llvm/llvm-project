; Use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

; Check that there is no overlap with unbounded array in different space

  ; Buffer<double> A[2] : register(t2, space4);
  ; Buffer<double> B : register(t20, space5);  // does not overlap
  ; Buffer<double> C[] : register(t2, space4); // overlaps with A

; CHECK: error: resource A at register 2 overlaps with resource C at register 2 in space 4
; CHECK-NOT: error: resource C at register 2 overlaps with resource B at register 20 in space 5

target triple = "dxil-pc-shadermodel6.3-library"

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
@C.str = private unnamed_addr constant [2 x i8] c"C\00", align 1

define void @test_not_overlapping_in_different_spaces() {
entry:

  ; Buffer<double> A[2] : register(t2, space4);
  %h0 = call target("dx.TypedBuffer", double, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 2, i32 10, ptr @A.str)

  ; Buffer<double> B : register(t20, space5);
  %h1 = call target("dx.TypedBuffer", i64, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 5, i32 20, i32 1, i32 0, ptr @B.str)

  ; Buffer<double> C[] : register(t2, space4);
  %h2 = call target("dx.TypedBuffer", double, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 -1, i32 10, ptr @C.str)

  ret void
}
