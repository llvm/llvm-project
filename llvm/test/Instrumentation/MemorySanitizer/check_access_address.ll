; RUN: opt < %s -msan-check-access-address=1 -S -passes=msan 2>&1 | FileCheck %s --implicit-check-not="call void @__msan_warning"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Test byval argument shadow alignment

define <2 x i64> @ByValArgumentShadowLargeAlignment(ptr byval(<2 x i64>) %p) sanitize_memory {
entry:
  %x = load <2 x i64>, ptr %p
  ret <2 x i64> %x
}

; CHECK-LABEL: @ByValArgumentShadowLargeAlignment
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 {{.*}}, ptr align 8 {{.*}}, i64 16, i1 false)
; CHECK: ret <2 x i64>


define i16 @ByValArgumentShadowSmallAlignment(ptr byval(i16) %p) sanitize_memory {
entry:
  %x = load i16, ptr %p
  ret i16 %x
}

; CHECK-LABEL: @ByValArgumentShadowSmallAlignment
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 2 {{.*}}, ptr align 2 {{.*}}, i64 2, i1 false)
; CHECK: ret i16


; Check instrumentation of stores. The check must precede the shadow store.

define void @Store(ptr nocapture %p, i32 %x) nounwind uwtable sanitize_memory {
entry:
  store i32 %x, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @Store
; CHECK: load {{.*}} @__msan_param_tls
; Shadow calculations must happen after the check.
; CHECK-NOT: xor
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: call void @__msan_warning_noreturn()
; CHECK: {{^[0-9]+}}:
; CHECK: xor
; CHECK: store
; CHECK: store i32 %x
; CHECK: ret void


