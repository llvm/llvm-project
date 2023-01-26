; RUN: opt < %s -msan-check-access-address=0                               -msan-eager-checks=1 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck --check-prefixes=CHECK,CONST %s --implicit-check-not=icmp --implicit-check-not="store i" --implicit-check-not="call void @__msan"
; RUN: opt < %s -msan-check-access-address=0 -msan-check-constant-shadow=0 -msan-eager-checks=1 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck --check-prefixes=CHECK       %s --implicit-check-not=icmp --implicit-check-not="store i" --implicit-check-not="call void @__msan"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test that returning a literal undef from main() triggers an MSan warning.

; main() is special: it inserts check for the return value
define i32 @main() nounwind uwtable sanitize_memory {
entry:
  ret i32 undef
}

; CHECK-LABEL: @main
; CHECK: store i32 0, ptr @__msan_retval_tls
; CONST: call void @__msan_warning_with_origin_noreturn
; CHECK: ret i32 undef


; This function stores known initialized value.
; Expect 2 stores: one for the shadow (0), one for the value (42), but no origin.
define void @StoreConstant(ptr nocapture %p) nounwind uwtable sanitize_memory {
entry:
  store i32 42, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @StoreConstant
; CHECK: store i32 0,
; CHECK: store i32 42,
; CHECK: ret void


; This function stores known uninitialized value.
; Expect 3 stores: shadow, value and origin.
; Expect no icmp(s): everything here is unconditional.
define void @StoreUndef(ptr nocapture %p) nounwind uwtable sanitize_memory {
entry:
  store i32 undef, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @StoreUndef
; CHECK: store i32 -1,
; CONST: store i32 0,
; CHECK: store i32 undef,
; CHECK: ret void


; This function stores known initialized value, but msan can't prove this.
define i32 @MaybeUninitialized(<2 x i64> noundef %acc) nounwind uwtable sanitize_memory {
entry:
  %shift = shufflevector <2 x i64> %acc, <2 x i64> poison, <2 x i32> <i32 1, i32 undef>
  %0 = add <2 x i64> %shift, %acc
  %1 = bitcast <2 x i64> %0 to <4 x i32>
  %conv = extractelement <4 x i32> %1, i64 0
  ret i32 %conv
}

; CHECK-LABEL: @MaybeUninitialized
; CHECK: store i32 extractelement (<4 x i32> bitcast (<2 x i64> <i64 0, i64 undef> to <4 x i32>), i64 0), ptr @__msan_retval_tls, align 8
; CHECK: store i32 0, ptr @__msan_retval_origin_tls

; This function stores known initialized value, but msan can't prove this.
define noundef i32 @MaybeUninitializedRetNoUndef(<2 x i64> noundef %acc) nounwind uwtable sanitize_memory {
entry:
  %shift = shufflevector <2 x i64> %acc, <2 x i64> poison, <2 x i32> <i32 1, i32 undef>
  %0 = add <2 x i64> %shift, %acc
  %1 = bitcast <2 x i64> %0 to <4 x i32>
  %conv = extractelement <4 x i32> %1, i64 0
  ret i32 %conv
}

; CHECK-LABEL: @MaybeUninitializedRetNoUndef
; CONST: br i1 icmp ne (i32 extractelement (<4 x i32> bitcast (<2 x i64> <i64 0, i64 undef> to <4 x i32>), i64 0), i32 0)
; CONST: call void @__msan_warning_with_origin_noreturn

; CHECK: call void @__msan_init()
