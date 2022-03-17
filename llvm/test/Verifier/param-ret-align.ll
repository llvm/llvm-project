; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Incorrect alignment of return type to called function!
define dso_local void @foo() {
entry:
  call <8192 x float> @bar()
  ret void
}

declare dso_local <8192 x float> @bar()
