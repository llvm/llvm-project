; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

declare void @llvm.icall.branch.funnel(...)

define void @invalid(ptr %s, ...) {
  ; CHECK: llvm.icall.branch.funnel must be called with musttail
  call void (...) @llvm.icall.branch.funnel(ptr %s)
  ret void
}

define void @valid(ptr %s, ...) {
  musttail call void (...) @llvm.icall.branch.funnel(ptr %s, ...)
  ret void
}
