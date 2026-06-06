; RUN: opt -S -passes=inline < %s | FileCheck %s

declare ptr @llvm.experimental.deoptimize.p0(...)

; Make sure we do not add incompatible attributes (noalias, align) to the deoptimize call.

define ptr @callee_noalias(ptr %c) {
  %v2 = call ptr (...) @llvm.experimental.deoptimize.p0(i32 42 ) [ "deopt"(i32 1) ]
  ret ptr %v2
}

; CHECK-LABEL: caller_noalias
; CHECK: call void (...) @llvm.experimental.deoptimize.isVoid(i32 42) [ "deopt"(i32 2, i32 1) ]
define void @caller_noalias(ptr %c) {
entry:
  %v = call noalias align 8 ptr @callee_noalias(ptr %c)  [ "deopt"(i32 2) ]
  ret void
}
