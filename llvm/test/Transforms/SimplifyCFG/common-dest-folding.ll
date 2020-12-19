; RUN: opt < %s -simplifycfg -S | FileCheck %s

;CHECK: @foo
;CHECK: and i32 %c1, %k
;CHECK: icmp eq i32
;CHECK: and i32 %c2, %k
;CHECK: icmp eq i32
;CHECK: or i1
;CHECK: ret
define i32 @foo(i32 %k, i32 %c1, i32 %c2) {
  %1 = and i32 %c1, %k
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %8, label %3

; <label>:3                                       ; preds = %0
  %4 = and i32 %c2, %k
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %8, label %6

; <label>:6                                       ; preds = %3
  %7 = tail call i32 (...) @bar() nounwind
  br label %8

; <label>:8                                       ; preds = %3, %0, %6
  ret i32 undef
}

;CHECK: @conduse
;CHECK: shl i32 1, %c1
;CHECK-NEXT: shl i32 1, %c2
;CHECK-NEXT: and i32
;CHECK-NEXT: icmp eq i32
;CHECK-NEXT: and i32
;CHECK-NEXT: icmp eq i32
;CHECK: ret
define i32 @conduse(i32 %k, i32 %c1, i32 %c2) #0 {
bb:
  %tmp = shl i32 1, %c1
  %tmp4 = shl i32 1, %c2
  %tmp1 = and i32 %tmp, %k
  %tmp2 = icmp eq i32 %tmp1, 0
  br i1 %tmp2, label %bb9, label %bb3

bb3:                                              ; preds = %bb
  %tmp5 = and i32 %tmp4, %k
  %tmp6 = icmp eq i32 %tmp5, 0
  br i1 %tmp6, label %bb9, label %bb7

bb7:                                              ; preds = %bb3
  %tmp8 = tail call i32 (...) @bar() #1
  br label %bb9

bb9:                                              ; preds = %bb7, %bb3, %bb
  ret i32 undef
}

declare i32 @bar(...)
