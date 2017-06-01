; RUN: opt -newgvn -S < %s | FileCheck %s
; RUN: opt -passes=newgvn -S -o - %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

;; All the loads in this testcase are useless, but it requires understanding that repeated
;; stores of the same value do not change the memory state to eliminate them.

define i32 @foo(i32*, i32)  {
; CHECK-LABEL: @foo
  store i32 5, i32* %0, align 4
  %3 = icmp ne i32 %1, 0
  br i1 %3, label %4, label %7

; <label>:4:                                      ; preds = %2
; CHECK-NOT: load
  %5 = load i32, i32* %0, align 4
; CHECK-NOT: add
  %6 = add nsw i32 5, %5
  br label %7

; <label>:7:                                      ; preds = %4, %2
  %.0 = phi i32 [ %6, %4 ], [ 5, %2 ]
; CHECK: phi i32 [ 10, %4 ], [ 5, %2 ]
  store i32 5, i32* %0, align 4
; CHECK-NOT: icmp
  %8 = icmp ne i32 %1, 0
; CHECK: br i1 %3
  br i1 %8, label %9, label %12

; <label>:9:                                      ; preds = %7
; CHECK-NOT: load
  %10 = load i32, i32* %0, align 4
; CHECK: add nsw i32 %.0, 5
  %11 = add nsw i32 %.0, %10
  br label %12

; <label>:12:                                     ; preds = %9, %7
  %.1 = phi i32 [ %11, %9 ], [ %.0, %7 ]
  ret i32 %.1
}

;; This is similar to the above, but it is a conditional store of the same value
;; which requires value numbering MemoryPhi properly to resolve.
define i32 @foo2(i32*, i32)  {
; CHECK-LABEL: @foo2
  store i32 5, i32* %0, align 4
  %3 = icmp ne i32 %1, 0
  br i1 %3, label %4, label %7

; <label>:4:                                      ; preds = %2
; CHECK-NOT: load
  %5 = load i32, i32* %0, align 4
; CHECK-NOT: add
  %6 = add nsw i32 5, %5
  br label %8

; <label>:7:                                      ; preds = %2
  store i32 5, i32* %0, align 4
  br label %8

; <label>:8:                                      ; preds = %7, %4
; CHECK: phi i32 [ 10, %4 ], [ 5, %5 ]
  %.0 = phi i32 [ %6, %4 ], [ 5, %7 ]
; CHECK-NOT: icmp
  %9 = icmp ne i32 %1, 0
; CHECK: br i1 %3
  br i1 %9, label %10, label %13

; <label>:10:                                     ; preds = %8
; CHECK-NOT: load
  %11 = load i32, i32* %0, align 4
; CHECK: add nsw i32 %.0, 5
  %12 = add nsw i32 %.0, %11
  br label %13

; <label>:13:                                     ; preds = %10, %8
  %.1 = phi i32 [ %12, %10 ], [ %.0, %8 ]
  ret i32 %.1
}
