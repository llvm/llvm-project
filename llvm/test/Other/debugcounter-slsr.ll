; RUN: opt -passes=slsr -S -debug-counter=slsr-counter=1 < %s | FileCheck %s --check-prefix=SKIP
; RUN: opt -passes=slsr -S < %s | FileCheck %s --check-prefix=EXEC

; Test that poison-generating annotations are dropped only when an SLSR
; opportunity is executed.

define void @stride_is_2s(i32 %b, i32 %s) {
; SKIP-LABEL: @stride_is_2s(
; SKIP-NEXT: %s2 = shl nuw nsw i32 %s, 1
; SKIP-NEXT: %t1 = add nuw nsw i32 %b, %s2
; SKIP-NEXT: call void @foo(i32 %t1)
; SKIP-NEXT: %s4 = shl i32 %s, 2
; SKIP-NEXT: %t2 = add i32 %b, %s4
; SKIP-NEXT: call void @foo(i32 %t2)
; SKIP-NEXT: ret void
;
; EXEC-LABEL: @stride_is_2s(
; EXEC-NEXT: %s2 = shl i32 %s, 1
; EXEC-NEXT: %t1 = add i32 %b, %s2
; EXEC-NEXT: call void @foo(i32 %t1)
; EXEC-NEXT: [[BUMP:%.*]] = shl i32 %s, 1
; EXEC-NEXT: %t2 = add i32 %t1, [[BUMP]]
; EXEC-NEXT: call void @foo(i32 %t2)
; EXEC-NEXT: ret void
;
  %s2 = shl nuw nsw i32 %s, 1
  %t1 = add nuw nsw i32 %b, %s2
  call void @foo(i32 %t1)
  %s4 = shl i32 %s, 2
  %t2 = add i32 %b, %s4
  call void @foo(i32 %t2)
  ret void
}

declare void @foo(i32)
