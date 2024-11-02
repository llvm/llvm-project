; RUN: opt < %s -passes=tsan -S | FileCheck --check-prefixes=CHECK,CHECK-OPT %s
; RUN: opt < %s -passes=tsan -tsan-instrument-read-before-write -S | FileCheck %s --check-prefixes=CHECK,CHECK-UNOPT
; RUN: opt < %s -passes=tsan -tsan-compound-read-before-write -S | FileCheck %s --check-prefixes=CHECK,CHECK-COMPOUND
; RUN: opt < %s -passes=tsan -tsan-distinguish-volatile -tsan-compound-read-before-write -S | FileCheck %s --check-prefixes=CHECK,CHECK-COMPOUND-VOLATILE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @IncrementMe(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @IncrementMe
; CHECK-OPT-NOT: __tsan_read4
; CHECK-COMPOUND-NOT: __tsan_read4
; CHECK-UNOPT: __tsan_read4
; CHECK-OPT: __tsan_write4
; CHECK-UNOPT: __tsan_write4
; CHECK-COMPOUND: __tsan_read_write4
; CHECK: ret void

define void @IncrementMeWithCallInBetween(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  call void @foo()
  store i32 %inc, i32* %ptr, align 4
  ret void
}

; CHECK-LABEL: define void @IncrementMeWithCallInBetween
; CHECK: __tsan_read4
; CHECK: __tsan_write4
; CHECK: ret void

declare void @foo()

define void @VolatileLoad(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load volatile i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @VolatileLoad
; CHECK-COMPOUND-NOT: __tsan_read4
; CHECK-COMPOUND-VOLATILE: __tsan_volatile_read4
; CHECK-COMPOUND: __tsan_read_write4
; CHECK-COMPOUND-VOLATILE: __tsan_write4
; CHECK: ret void

define void @VolatileStore(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  store volatile i32 %inc, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @VolatileStore
; CHECK-COMPOUND-NOT: __tsan_read4
; CHECK-COMPOUND-VOLATILE: __tsan_read4
; CHECK-COMPOUND: __tsan_read_write4
; CHECK-COMPOUND-VOLATILE: __tsan_volatile_write4
; CHECK: ret void

define void @VolatileBoth(i32* nocapture %ptr) nounwind uwtable sanitize_thread {
entry:
  %0 = load volatile i32, i32* %ptr, align 4
  %inc = add nsw i32 %0, 1
  store volatile i32 %inc, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @VolatileBoth
; CHECK-COMPOUND-NOT: __tsan_read4
; CHECK-COMPOUND-VOLATILE: __tsan_volatile_read4
; CHECK-COMPOUND: __tsan_read_write4
; CHECK-COMPOUND-VOLATILE: __tsan_volatile_write4
; CHECK: ret void

