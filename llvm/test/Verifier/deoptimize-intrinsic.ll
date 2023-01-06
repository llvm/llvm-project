; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

declare i8 @llvm.experimental.deoptimize.i8(...)
declare void @llvm.experimental.deoptimize.isVoid(...)
declare cc40 void @llvm.experimental.deoptimize.double(...)

declare void @unknown()

define void @f_notail() {
entry:
  call void(...) @llvm.experimental.deoptimize.isVoid(i32 0) [ "deopt"() ]
; CHECK: calls to experimental_deoptimize must be followed by a return
  call void @unknown()
  ret void
}

define void @f_nodeopt() {
entry:
  call void(...) @llvm.experimental.deoptimize.isVoid()
; CHECK: experimental_deoptimize must have exactly one "deopt" operand bundle
  ret void
}

define void @f_invoke() personality i8 3 {
entry:
  invoke void(...) @llvm.experimental.deoptimize.isVoid(i32 0, float 0.0) to label %ok unwind label %not_ok
; CHECK: experimental_deoptimize cannot be invoked

ok:
  ret void

not_ok:
  %0 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  ret void
}

define i8 @f_incorrect_return() {
entry:
  %val = call i8(...) @llvm.experimental.deoptimize.i8() [ "deopt"() ]
; CHECK: calls to experimental_deoptimize must be followed by a return of the value computed by experimental_deoptimize
  ret i8 0
}

; CHECK: All llvm.experimental.deoptimize declarations must have the same calling convention
