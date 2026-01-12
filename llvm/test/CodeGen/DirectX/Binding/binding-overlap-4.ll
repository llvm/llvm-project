; Use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

; Check multiple overlap errors.

; A overlaps with B
; A overlaps with C
; B overlaps with C
; StructuredBuffer<float> A[5] : register(t1); // 1-5
; StructuredBuffer<float> B[2] : register(t2); // 2-3
; StructuredBuffer<float> C[3] : register(t3); // 3-5 

; CHECK: error: resource A at register 1 overlaps with resource B at register 2 in space 0
; CHECK: error: resource A at register 1 overlaps with resource C at register 3 in space 0
; CHECK: error: resource B at register 2 overlaps with resource C at register 3 in space 0

target triple = "dxil-pc-shadermodel6.3-library"

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
@C.str = private unnamed_addr constant [2 x i8] c"C\00", align 1

; Fake globals to store handles in; this is to make sure the handlefrombinding calls
; are not optimized away by llc.
@One = internal global { target("dx.RawBuffer", float, 0, 0) } poison, align 4
@Two = internal global { target("dx.RawBuffer", float, 0, 0) } poison, align 4
@Three = internal global { target("dx.RawBuffer", float, 0, 0) } poison, align 4

define void @test_overlapping() "hlsl.export" {
entry:
  %h1 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 5, i32 0, ptr @A.str)
  store target("dx.RawBuffer", float, 0, 0) %h1, ptr @One, align 4
  
  %h2 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 2, i32 0, ptr @B.str)
  store target("dx.RawBuffer", float, 0, 0) %h2, ptr @Two, align 4
 
  %h3 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 3, i32 4, ptr @C.str)
  store target("dx.RawBuffer", float, 0, 0) %h3, ptr @Three, align 4
 
  ret void
}
