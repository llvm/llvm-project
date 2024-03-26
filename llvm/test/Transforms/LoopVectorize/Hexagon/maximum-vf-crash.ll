; RUN: opt -mtriple=hexagon -hexagon-autohvx -passes=loop-vectorize -S < %s 2>&1 | FileCheck %s

; Check that we don't crash.

; CHECK-LABEL: @f
; CHECK: vector.body

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"

; Function Attrs: optsize
define i32 @f() #0 {
entry:
  br label %loop

loop:
  %g.016 = phi i32 [ 0, %entry ], [ %g.1.lcssa, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %0 = load i8, ptr undef, align 1
  %g.1.lcssa = add i32 %g.016, undef
  %iv.next = add nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret i32 %g.1.lcssa
}

attributes #0 = { optsize "target-features"="+hvx-length128b" }
