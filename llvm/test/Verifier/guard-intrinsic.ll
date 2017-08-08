; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

declare void @unknown()

define void @f_nodeopt() {
entry:
  call void(i1, ...) @llvm.experimental.guard(i1 undef, i32 1, i32 2)
; CHECK: guard must have exactly one "deopt" operand bundle
  ret void
}

define void @f_invoke() personality i8 3 {
entry:
  invoke void(i1, ...) @llvm.experimental.guard(i1 undef, i32 0, float 0.0) [ "deopt"() ] to label %ok unwind label %not_ok
; CHECK: guard cannot be invoked

ok:
  ret void

not_ok:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  ret void
}
