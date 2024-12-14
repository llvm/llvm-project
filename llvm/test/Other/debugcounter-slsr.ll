; REQUIRES: asserts
; RUN: opt -passes=slsr,gvn -S -debug-counter=slsr-counter=1  < %s | FileCheck %s

;; Test that, with debug counters on, we will skip the first slsr opportunity.

define void @stride_is_2s(i32 %b, i32 %s) {
; CHECK-LABEL: @stride_is_2s(
; CHECK-NEXT:    %s2 = shl i32 %s, 1
; CHECK-NEXT:    %t1 = add i32 %b, %s2
; CHECK-NEXT:    call void @foo(i32 %t1)
; CHECK-NEXT:    %t2 = add i32 %t1, %s2
; CHECK-NEXT:    call void @foo(i32 %t2)
; CHECK-NEXT:    %s6 = mul i32 %s, 6
; CHECK-NEXT:    %t3 = add i32 %b, %s6
; CHECK-NEXT:    call void @foo(i32 %t3)
; CHECK-NEXT:    ret void
;
  %s2 = shl i32 %s, 1
  %t1 = add i32 %b, %s2
  call void @foo(i32 %t1)
  %s4 = shl i32 %s, 2
  %t2 = add i32 %b, %s4
  call void @foo(i32 %t2)
  %s6 = mul i32 %s, 6
  %t3 = add i32 %b, %s6
  call void @foo(i32 %t3)
  ret void
}

declare void @foo(i32)