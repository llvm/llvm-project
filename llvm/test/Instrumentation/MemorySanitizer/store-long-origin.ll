; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S          \
; RUN: -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Test origin for longer stores.

define void @Store8(i64* nocapture %p, i64 %x) sanitize_memory {
entry:
  store i64 %x, i64* %p, align 8
  ret void
}

; Single 8-byte origin store
; CHECK-LABEL: define void @Store8(
; CHECK: store i64 {{.*}}, align 8
; CHECK: store i64 {{.*}}, align 8
; CHECK: store i64 {{.*}}, align 8
; CHECK: ret void

define void @Store8_align4(i64* nocapture %p, i64 %x) sanitize_memory {
entry:
  store i64 %x, i64* %p, align 4
  ret void
}

; Two 4-byte origin stores
; CHECK-LABEL: define void @Store8_align4(
; CHECK: store i64 {{.*}}, align 4
; CHECK: store i32 {{.*}}, align 4
; CHECK: getelementptr i32, i32* {{.*}}, i32 1
; CHECK: store i32 {{.*}}, align 4
; CHECK: store i64 {{.*}}, align 4
; CHECK: ret void

%struct.S = type { i32, i32, i32 }

define void @StoreAgg(%struct.S* nocapture %p, %struct.S %x) sanitize_memory {
entry:
  store %struct.S %x, %struct.S* %p, align 4
  ret void
}

; Three 4-byte origin stores
; CHECK-LABEL: define void @StoreAgg(
; CHECK: store { i32, i32, i32 }  {{.*}}, align 4
; CHECK: store i32 {{.*}}, align 4
; CHECK: getelementptr i32, i32* {{.*}}, i32 1
; CHECK: store i32 {{.*}}, align 4
; CHECK: getelementptr i32, i32* {{.*}}, i32 2
; CHECK: store i32 {{.*}}, align 4
; CHECK: store %struct.S {{.*}}, align 4
; CHECK: ret void


define void @StoreAgg8(%struct.S* nocapture %p, %struct.S %x) sanitize_memory {
entry:
  store %struct.S %x, %struct.S* %p, align 8
  ret void
}

; 8-byte + 4-byte origin stores
; CHECK-LABEL: define void @StoreAgg8(
; CHECK: store { i32, i32, i32 }  {{.*}}, align 8
; CHECK: store i64 {{.*}}, align 8
; CHECK: getelementptr i32, i32* {{.*}}, i32 2
; CHECK: store i32 {{.*}}, align 8
; CHECK: store %struct.S {{.*}}, align 8
; CHECK: ret void


%struct.Q = type { i64, i64, i64 }
define void @StoreAgg24(%struct.Q* nocapture %p, %struct.Q %x) sanitize_memory {
entry:
  store %struct.Q %x, %struct.Q* %p, align 8
  ret void
}

; 3 8-byte origin stores
; CHECK-LABEL: define void @StoreAgg24(
; CHECK: store { i64, i64, i64 }  {{.*}}, align 8
; CHECK: store i64 {{.*}}, align 8
; CHECK: getelementptr i64, i64* {{.*}}, i32 1
; CHECK: store i64 {{.*}}, align 8
; CHECK: getelementptr i64, i64* {{.*}}, i32 2
; CHECK: store i64 {{.*}}, align 8
; CHECK: store %struct.Q {{.*}}, align 8
; CHECK: ret void
