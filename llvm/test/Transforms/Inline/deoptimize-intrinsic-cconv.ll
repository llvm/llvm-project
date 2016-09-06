; RUN: opt -S -always-inline < %s | FileCheck %s

declare cc42 i32 @llvm.experimental.deoptimize.i32(...)

define i32 @callee_with_coldcc() alwaysinline {
  %v0 = call cc42 i32(...) @llvm.experimental.deoptimize.i32(i32 1) [ "deopt"() ]
  ret i32 %v0
}

define void @caller_with_coldcc() {
; CHECK-LABEL: @caller_with_coldcc(
; CHECK-NEXT:  call cc42 void (...) @llvm.experimental.deoptimize.isVoid(i32 1) [ "deopt"() ]
; CHECK-NEXT:  ret void

  %val = call i32 @callee_with_coldcc()
  ret void
}

; CHECK: declare cc42 void @llvm.experimental.deoptimize.isVoid(...)
