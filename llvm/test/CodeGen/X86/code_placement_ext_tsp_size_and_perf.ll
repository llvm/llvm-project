; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -verify-machineinstrs -enable-ext-tsp-block-placement -apply-ext-tsp-for-size < %s | FileCheck %s

;; Cold function with optsize: should use ext-tsp for size.
;; The size-optimized layout keeps the original order (b0, b1, b2) since it
;; minimizes code size by avoiding extra jumps.
define void @cold_func() optsize !prof !2 {
;
; +-----+
; | b0  | -+
; +-----+  |
;   |      |
;   | 10   |
;   v      |
; +-----+  |
; | b1  |  | 10000
; +-----+  |
;   |      |
;   | 10   |
;   v      |
; +-----+  |
; | b2  | <+
; +-----+
;
; CHECK-LABEL: cold_func:
; CHECK: %b0
; CHECK: %b1
; CHECK: %b2

b0:
  %call = call zeroext i1 @a()
  br i1 %call, label %b1, label %b2, !prof !1

b1:
  call void @d()
  call void @d()
  call void @d()
  br label %b2

b2:
  call void @e()
  ret void
}

;; Hot function without optsize: should use ext-tsp for perf.
;; The perf-optimized layout reorders blocks to place the most likely successor
;; (b2) right after b0, giving b0 -> b2 -> b1.
define void @hot_func() !prof !3 {
;
; +-----+
; | b0  | -+
; +-----+  |
;   |      |
;   | 40   |
;   v      |
; +-----+  |
; | b1  |  | 100
; +-----+  |
;   |      |
;   | 40   |
;   v      |
; +-----+  |
; | b2  | <+
; +-----+
;
; CHECK-LABEL: hot_func:
; CHECK: %b0
; CHECK: %b2
; CHECK: %b1

b0:
  %call = call zeroext i1 @a()
  br i1 %call, label %b1, label %b2, !prof !1

b1:
  call void @d()
  call void @d()
  call void @d()
  br label %b2

b2:
  call void @e()
  ret void
}

declare zeroext i1 @a()
declare void @d()
declare void @e()

!1 = !{!"branch_weights", i32 10, i32 10000}
!2 = !{!"function_entry_count", i64 1}
!3 = !{!"function_entry_count", i64 2200}
