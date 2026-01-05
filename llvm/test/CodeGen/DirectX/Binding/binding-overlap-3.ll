; Use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

; Check multiple overlap errors.
; Also check different resource class with same binding values is ok (no error expected).

; C overlaps with A
; C overlaps with B
; StructuredBuffer<float> A : register(t5);
; StructuredBuffer<float> B : register(t9);
; StructuredBuffer<float> C[10] : register(t0);
; RWBuffer<float> S[10] : register(u0);

; CHECK: error: resource C at register 0 overlaps with resource A at register 5 in space 0
; CHECK: error: resource C at register 0 overlaps with resource B at register 9 in space 0

target triple = "dxil-pc-shadermodel6.3-library"

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
@C.str = private unnamed_addr constant [2 x i8] c"C\00", align 1
@S.str = private unnamed_addr constant [2 x i8] c"S\00", align 1

; Fake globals to store handles in; this is to make sure the handlefrombinding calls
; are not optimized away by llc.
@One = internal global { target("dx.RawBuffer", float, 0, 0) } poison, align 4
@Two = internal global { target("dx.RawBuffer", float, 0, 0) } poison, align 4
@Three = internal global { target("dx.RawBuffer", float, 0, 0) } poison, align 4
@Four = internal global { target("dx.TypedBuffer", float, 1, 0, 0) } poison, align 4

define void @test_overlapping() "hlsl.export" {
entry:
  %h1 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr @A.str)
  store target("dx.RawBuffer", float, 0, 0) %h1, ptr @One, align 4
  
  %h2 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 9, i32 1, i32 0, ptr @B.str)
  store target("dx.RawBuffer", float, 0, 0) %h2, ptr @Two, align 4
 
  %h3 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 4, ptr @C.str)
  store target("dx.RawBuffer", float, 0, 0) %h3, ptr @Three, align 4
 
  %h4 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @S.str)
  store target("dx.TypedBuffer", float, 1, 0, 0) %h4, ptr @Four, align 4
  
  ret void
}
