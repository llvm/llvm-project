; RUN: opt < %s -msan-check-access-address=0 -msan-check-constant-shadow=1     \
; RUN: -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test that returning a literal undef from main() triggers an MSan warning.

; main() is special: it inserts check for the return value
define i32 @main() nounwind uwtable sanitize_memory {
entry:
  ret i32 undef
}

; CHECK-LABEL: @main
; CHECK: call void @__msan_warning_with_origin_noreturn
; CHECK: ret i32 undef


; This function stores known initialized value.
; Expect 2 stores: one for the shadow (0), one for the value (42), but no origin.
define void @StoreConstant(i32* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  store i32 42, i32* %p, align 4
  ret void
}

; CHECK-LABEL: @StoreConstant
; CHECK-NOT: store i32
; CHECK: store i32 0,
; CHECK-NOT: store i32
; CHECK: store i32 42,
; CHECK-NOT: store i32
; CHECK: ret void


; This function stores known uninitialized value.
; Expect 3 stores: shadow, value and origin.
; Expect no icmp(s): everything here is unconditional.
define void @StoreUndef(i32* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  store i32 undef, i32* %p, align 4
  ret void
}

; CHECK-LABEL: @StoreUndef
; CHECK-NOT: icmp
; CHECK: store i32
; CHECK-NOT: icmp
; CHECK: store i32
; CHECK-NOT: icmp
; CHECK: store i32
; CHECK-NOT: icmp
; CHECK: ret void
