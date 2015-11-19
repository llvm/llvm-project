; RUN: opt < %s -S -scalarrepl | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.1"

; CHECK: test
; CHECK-NOT: alloca

define void @test() nounwind {
entry:
  %a156286 = alloca [4 x <4 x float>], align 16
  br i1 undef, label %cif_done, label %for_test158.preheader

for_test158.preheader:                            ; preds = %entry
  %a156286305 = bitcast [4 x <4 x float>]* %a156286 to i8*
  call void @llvm.memset.p0i8.i64(i8* %a156286305, i8 -1, i64 64, i32 16, i1 false)
  unreachable

cif_done:                                         ; preds = %entry
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
