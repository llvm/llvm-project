; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: invalid uses of intrinsic global variable
; CHECK-NEXT: ptr @llvm.used
@llvm.used = appending global [1 x ptr] [ptr @foo]

; CHECK: invalid uses of intrinsic global variable
; CHECK-NEXT: ptr @llvm.compiler.used
@llvm.compiler.used = appending global [1 x ptr] [ptr @bar]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

@used_user = global ptr @llvm.used
@compiler_used_user = global ptr @llvm.compiler.used
