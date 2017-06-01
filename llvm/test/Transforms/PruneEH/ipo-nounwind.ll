; RUN: opt -S -prune-eh < %s | FileCheck %s

declare void @may_throw()

; @callee below may be an optimized form of this function, which can
; throw at runtime (see r265762 for more details):
; 
; define linkonce_odr void @callee(i32* %ptr) noinline {
; entry:
;   %val0 = load atomic i32, i32* %ptr unordered, align 4
;   %val1 = load atomic i32, i32* %ptr unordered, align 4
;   %cmp = icmp eq i32 %val0, %val1
;   br i1 %cmp, label %left, label %right

; left:
;   ret void

; right:
;   call void @may_throw()
;   ret void
; }

define linkonce_odr void @callee(i32* %ptr) noinline {
  ret void
}

define i32 @caller(i32* %ptr) personality i32 3 {
; CHECK-LABEL: @caller(
; CHECK:  invoke void @callee(i32* %ptr)
; CHECK-NEXT:          to label %normal unwind label %unwind

entry:
  invoke void @callee(i32* %ptr)
          to label %normal unwind label %unwind

normal:
  ret i32 1

unwind:
  %res = landingpad { i8*, i32 }
         cleanup
  ret i32 2
}
