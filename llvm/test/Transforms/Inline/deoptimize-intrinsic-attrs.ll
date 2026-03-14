; RUN: opt -S -passes=always-inline < %s | FileCheck %s

declare cc42 i32 @llvm.experimental.deoptimize.i32(...)

define i32 @callee_with_coldcc() alwaysinline {
  %v0 = call cc42 i32(...) @llvm.experimental.deoptimize.i32(i32 1) #0 [ "deopt"() ]
  ret i32 %v0
}

define void @caller_with_coldcc() {
; CHECK-LABEL: @caller_with_coldcc(
; CHECK-NEXT:  call cc42 void (...) @llvm.experimental.deoptimize.isVoid(i32 1) #1 [ "deopt"() ]
; CHECK-NEXT:  ret void

  %val = call i32 @callee_with_coldcc()
  ret void
}

attributes #0 = { "deopt-lowering"="live-in" }

; CHECK: declare cc42 void @llvm.experimental.deoptimize.isVoid(...)
; CHECK: attributes #1 = { "deopt-lowering"="live-in" }
