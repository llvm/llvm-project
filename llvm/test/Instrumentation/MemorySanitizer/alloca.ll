; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s --check-prefixes=CHECK,INLINE
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-poison-stack-with-call=1 -S | FileCheck %s --check-prefixes=CHECK,CALL
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,ORIGIN
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck %s --check-prefixes=CHECK,ORIGIN

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @static() sanitize_memory {
entry:
  %x = alloca i32, align 4
  ret void
}

; CHECK-LABEL: define void @static(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin4(i8* {{.*}}, i64 4,
; CHECK: ret void


define void @dynamic() sanitize_memory {
entry:
  br label %l
l:
  %x = alloca i32, align 4
  ret void
}

; CHECK-LABEL: define void @dynamic(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin4(i8* {{.*}}, i64 4,
; CHECK: ret void

define void @array() sanitize_memory {
entry:
  %x = alloca i32, i64 5, align 4
  ret void
}

; CHECK-LABEL: define void @array(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 20, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 20)
; ORIGIN: call void @__msan_set_alloca_origin4(i8* {{.*}}, i64 20,
; CHECK: ret void

define void @array_non_const(i64 %cnt) sanitize_memory {
entry:
  %x = alloca i32, i64 %cnt, align 4
  ret void
}

; CHECK-LABEL: define void @array_non_const(
; CHECK: %[[A:.*]] = mul i64 4, %cnt
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 %[[A]], i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 %[[A]])
; ORIGIN: call void @__msan_set_alloca_origin4(i8* {{.*}}, i64 %[[A]],
; CHECK: ret void
