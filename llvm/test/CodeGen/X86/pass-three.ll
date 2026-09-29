; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.3.0"


define { ptr, i64, ptr } @copy_3(ptr %a, i64 %b, ptr %c) nounwind {
entry:
  %0 = insertvalue { ptr, i64, ptr } undef, ptr %a, 0
  %1 = insertvalue { ptr, i64, ptr } %0, i64 %b, 1
  %2 = insertvalue { ptr, i64, ptr } %1, ptr %c, 2
  ret { ptr, i64, ptr } %2
}

; CHECK-LABEL: copy_3:
; CHECK-NOT: (%rdi)
; CHECK: ret
