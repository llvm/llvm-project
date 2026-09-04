; A pseudo-probe must not be inserted between a musttail or
; llvm.experimental.deoptimize call and its following return, as this produces
; invalid IR. Verify that probes are inserted before these calls instead.
;
; RUN: opt < %s -passes=pseudo-probe -S | FileCheck %s

; CHECK-LABEL: define i32 @mt(ptr %p)
; CHECK:      call void @llvm.pseudoprobe
; CHECK-NEXT: %v = musttail call i32 @callee(ptr %p)
; CHECK-NEXT: ret i32 %v
define i32 @mt(ptr %p) {
  %v = musttail call i32 @callee(ptr %p)
  ret i32 %v
}

; CHECK-LABEL: define i32 @deopt()
; CHECK:      call void @llvm.pseudoprobe
; CHECK-NEXT: %v = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"() ]
; CHECK-NEXT: ret i32 %v
define i32 @deopt() {
  %v = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"() ]
  ret i32 %v
}

declare i32 @callee(ptr)
declare i32 @llvm.experimental.deoptimize.i32(...)
