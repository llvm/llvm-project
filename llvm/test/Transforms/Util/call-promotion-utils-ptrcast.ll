; RUN: opt -S -passes=pgo-icall-prom -icp-total-percent-threshold=0 -icp-max-prom=4 < %s 2>&1 | FileCheck %s

; Test that CallPromotionUtils will promote calls which require pointer casts.

@foo = common global ptr null, align 8

; Check ptrcast arguments.
define i64 @func1(ptr %a) {
  ret i64 undef
}

; Check ptrcast return.
define ptr @func2(i64 %a) {
  ret ptr undef
}

; Check ptrcast arguments and return.
define ptr @func3(ptr %a) {
  ret ptr undef
}

; Check mixed ptrcast and bitcast.
define ptr @func4(double %f) {
  ret ptr undef
}

define i64 @bar() {
  %tmp = load ptr, ptr @foo, align 8

; CHECK: [[ARG:%[0-9]+]] = bitcast i64 1 to double
; CHECK-NEXT: [[RET:%[0-9]+]] = call ptr @func4(double [[ARG]])
; CHECK-NEXT: ptrtoint ptr [[RET]] to i64

; CHECK: [[RET:%[0-9]+]] = call ptr @func2(i64 1)
; CHECK-NEXT: ptrtoint ptr [[RET]] to i64

; CHECK: [[ARG:%[0-9]+]] = inttoptr i64 1 to ptr
; CHECK-NEXT: [[RET:%[0-9]+]] = call ptr @func3(ptr [[ARG]])
; CHECK-NEXT: ptrtoint ptr [[RET]] to i64

; CHECK: [[ARG:%[0-9]+]] = inttoptr i64 1 to ptr
; CHECK-NEXT: call i64 @func1(ptr [[ARG]])
; CHECK-NOT: ptrtoint
; CHECK-NOT: bitcast

  %call = call i64 %tmp(i64 1), !prof !1
  ret i64 %call
}

!1 = !{!"VP", i32 0, i64 1600, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410, i64 -6929281286627296573, i64 150, i64 -2545542355363006406, i64 10}
