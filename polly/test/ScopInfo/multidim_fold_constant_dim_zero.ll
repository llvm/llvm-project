; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-scops -analyze -debug -S < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; This test case at some point crashed Polly due to a 'division by zero'
; when trying to fold the constant dimension into outer dimension.
; We verify that this scop is detected without crash. We also test the
; output to undertand that the scop has been analyzed, but has also been
; invalidated due to the zero size dimension.

; CHECK: Assumed Context:
; CHECK-NEXT: {  : false }
; CHECK-NEXT: Invalid Context:
; CHECK-NEXT: {  : false }
; CHECK:      Arrays {
; CHECK-NEXT:     i8 MemRef_arg[*][0]; // Element size 1
; CHECK-NEXT: }
; CHECK-NEXT: Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     i8 MemRef_arg[*][ { [] -> [(0)] } ]; // Element size 1
; CHECK-NEXT: }
; CHECK-NEXT: Alias Groups (0):
; CHECK-NEXT:     n/a
; CHECK-NEXT: Statements {
; CHECK-NEXT: 	Stmt_bb2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb2[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb2[] -> [] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb2[] -> MemRef_arg[0, 0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb2[] -> MemRef_arg[o0, o1] : false };
; CHECK-NEXT: }

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #0

define void @hoge([0 x [0 x i8]]* noalias %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  br i1 false, label %bb5, label %bb2

bb2:                                              ; preds = %bb1
  %tmp = getelementptr [0 x [0 x i8]], [0 x [0 x i8]]* %arg, i64 0, i64 0, i64 0
  store i8 32, i8* %tmp, align 1
  %tmp3 = getelementptr [0 x [0 x i8]], [0 x [0 x i8]]* %arg, i64 0, i64 0, i64 0
  %tmp4 = getelementptr i8, i8* %tmp3, i64 1
  tail call void @llvm.memset.p0i8.i64(i8* %tmp4, i8 32, i64 0, i32 1, i1 false)
  br label %bb5

bb5:                                              ; preds = %bb2, %bb1
  br i1 undef, label %bb6, label %bb1

bb6:                                              ; preds = %bb5
  ret void
}
