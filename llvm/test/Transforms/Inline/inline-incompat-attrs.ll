; RUN: opt < %s -passes=inline -inline-threshold=100 -S | FileCheck %s

;; caller1/caller2/callee1/callee2 test functions with incompatible attributes
;; won't be inlined into each other.

define i32 @callee1(i32 %x) {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  call void @extern()
  ret i32 %x3
}

define i32 @callee2(i32 %x) #0 {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  call void @extern()
  ret i32 %x3
}

define i32 @caller1(i32 %y1) {
;; caller1 doesn't have use-sample-profile attribute but callee2 has,
;; so callee2 won't be inlined into caller1.
;; caller1 and callee1 don't have use-sample-profile attribute, so
;; callee1 can be inlined into caller1.
; CHECK-LABEL: @caller1(
; CHECK: call i32 @callee2
; CHECK-NOT: call i32 @callee1
  %y2 = call i32 @callee2(i32 %y1)
  %y3 = call i32 @callee1(i32 %y2)
  ret i32 %y3
}

define i32 @caller2(i32 %y1) #0 {
;; caller2 and callee2 both have use-sample-profile attribute, so
;; callee2 can be inlined into caller2.
;; caller2 has use-sample-profile attribute but callee1 doesn't have,
;; so callee1 won't be inlined into caller2.
; CHECK-LABEL: @caller2(
; CHECK-NOT: call i32 @callee2
; CHECK: call i32 @callee1
  %y2 = call i32 @callee2(i32 %y1)
  %y3 = call i32 @callee1(i32 %y2)
  ret i32 %y3
}

declare void @extern()

attributes #0 = { "use-sample-profile" }
