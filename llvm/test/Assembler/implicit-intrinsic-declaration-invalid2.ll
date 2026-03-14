; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Use of intrinsic as non-callee should be rejected.

; CHECK: error: intrinsic can only be used as callee
define void @test() {
  call void @foo(ptr @llvm.umax)
  ret void
}

declare void @foo(ptr)
