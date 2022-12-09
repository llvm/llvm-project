; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Test that MemorySanitizer correctly handles unreachable blocks.

define i32 @Func(ptr %p) nounwind uwtable {
entry:
  br label %exit

unreachable:
  %x = load i32, ptr %p
  br label %exit

exit:
  %z = phi i32 [ 42, %entry ], [ %x, %unreachable ]
  ret i32 %z
}

; CHECK-LABEL: @Func
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32 42


define i32 @UnreachableLoop() nounwind uwtable {
entry:
  ret i32 0

zzz:
  br label %xxx

xxx:
  br label %zzz
}

; CHECK-LABEL: @UnreachableLoop
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32 0
