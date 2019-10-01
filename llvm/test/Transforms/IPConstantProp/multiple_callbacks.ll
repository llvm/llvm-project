; RUN: opt -ipconstprop -S < %s | FileCheck %s
;
;
;                            /---------------------------------------|
;                            |                /----------------------|----|
;                            |                |                /-----|    |
;                            V                V                V     |    |
;    void broker(int (*cb0)(int), int (*cb1)(int), int (*cb2)(int), int, int);
;
;    static int cb0(int zero) {
;      return zero;
;    }
;    static int cb1(int unknown) {
;      return unknown;
;    }
;    static int cb2(int unknown) {
;      cb0(0);
;      return unknown;
;    }
;    static int cb3(int unknown) {
;      return unknown;
;    }
;    static int cb4(int unknown) {
;      return unknown;
;    }
;
;    void foo() {
;      cb0(0);
;      cb3(1);
;      broker(cb0, cb1, cb0, 0, 1);
;      broker(cb1, cb2, cb2, 0, 1);
;      broker(cb3, cb2, cb3, 0, 1);
;      broker(cb4, cb4, cb4, 0, 1);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define internal i32 @cb0(i32 %zero) {
entry:
; CHECK:      @cb0
; CHECK-NEXT: entry
; CHECK-NEXT: ret i32 0
  ret i32 %zero
}

define internal i32 @cb1(i32 %unknown) {
entry:
; CHECK: ret i32 %unknown
  ret i32 %unknown
}

define internal i32 @cb2(i32 %unknown) {
entry:
  %call = call i32 @cb0(i32 0)
; CHECK: ret i32 %unknown
  ret i32 %unknown
}

define internal i32 @cb3(i32 %unknown) {
entry:
; CHECK: ret i32 %unknown
  ret i32 %unknown
}

define internal i32 @cb4(i32 %unknown) {
entry:
; CHECK: ret i32 %unknown
  ret i32 %unknown
}

define void @foo() {
entry:
  %call = call i32 @cb0(i32 0)
  %call1 = call i32 @cb3(i32 1)
  call void @broker(i32 (i32)* nonnull @cb0, i32 (i32)* nonnull @cb1, i32 (i32)* nonnull @cb0, i32 0, i32 1)
  call void @broker(i32 (i32)* nonnull @cb1, i32 (i32)* nonnull @cb2, i32 (i32)* nonnull @cb2, i32 0, i32 1)
  call void @broker(i32 (i32)* nonnull @cb3, i32 (i32)* nonnull @cb2, i32 (i32)* nonnull @cb3, i32 0, i32 1)
  call void @broker(i32 (i32)* nonnull @cb4, i32 (i32)* nonnull @cb4, i32 (i32)* nonnull @cb4, i32 0, i32 1)
  ret void
}

declare !callback !3 void @broker(i32 (i32)*, i32 (i32)*, i32 (i32)*, i32, i32)

!0 = !{i64 0, i64 3, i1 false}
!1 = !{i64 1, i64 4, i1 false}
!2 = !{i64 2, i64 3, i1 false}
!3 = !{!0, !2, !1}
