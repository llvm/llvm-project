; RUN: opt < %s -S -passes=early-cse | FileCheck %s

;When icmp's argument is a volatile instruction, we do not assume that icmp can
;be simplified.
define dso_local noundef i32 @_Z1funcv() {
; CHECK-LABEL: @_Z1funcv(
; CHECK-NEXT:    %1 = alloca ptr, align 8
; CHECK-NEXT:    store volatile ptr null, ptr %1, align 8
; CHECK-NEXT:    %2 = load volatile ptr, ptr %1, align 8
; CHECK-NEXT:    %3 = ptrtoint ptr %2 to i64
; CHECK-NEXT:    %4 = and i64 %3, 1
; CHECK-NEXT:    %5 = icmp eq i64 %4, 0
; CHECK-NEXT:    %6 = zext i1 %5 to i32
; CHECK-NEXT:    ret i32 %6
;
  %1 = alloca ptr, align 8
  store volatile ptr null, ptr %1, align 8
  %2 = load volatile ptr, ptr %1, align 8
  %3 = ptrtoint ptr %2 to i64
  %4 = and i64 %3, 1
  %5 = icmp eq i64 %4, 0
  %6 = zext i1 %5 to i32
  ret i32 %6
}

define dso_local noundef i32 @_Z2funcv() {
; CHECK-LABEL: @_Z2funcv(
; CHECK-NEXT:    %1 = alloca ptr, align 8
; CHECK-NEXT:    store ptr null, ptr %1, align 8
; CHECK-NEXT:    ret i32 1
;
  %1 = alloca ptr, align 8
  store ptr null, ptr %1, align 8
  %2 = load ptr, ptr %1, align 8
  %3 = ptrtoint ptr %2 to i64
  %4 = and i64 %3, 1
  %5 = icmp eq i64 %4, 0
  %6 = zext i1 %5 to i32
  ret i32 %6
}
