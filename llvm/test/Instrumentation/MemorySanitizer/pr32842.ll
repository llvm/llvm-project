; Regression test for https://bugs.llvm.org/show_bug.cgi?id=32842
;
; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s
;target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define zeroext i1 @_Z1fii(i32 %x, i32 %y) sanitize_memory {
entry:
  %cmp = icmp slt i32 %x, %y
  ret i1 %cmp
}

; CHECK:      [[X:[^ ]+]] = load{{.*}}__msan_param_tls{{.*}}
; CHECK:      [[Y:[^ ]+]] = load{{.*}}__msan_param_tls{{.*}}
; CHECK:      [[OR:[^ ]+]] = or i32 [[X]], [[Y]]

; Make sure the shadow of the (x < y) comparison isn't truncated to i1.
; CHECK-NOT:  trunc i32 [[OR]] to i1
; CHECK:      [[CMP:[^ ]+]] = icmp ne i32 [[OR]], 0
; CHECK:      store i1 [[CMP]],{{.*}}__msan_retval_tls
