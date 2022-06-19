; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Incorrect alignment of argument passed to called function!
define dso_local void @foo(<8192 x float> noundef %vec) {
entry:
  call void @bar(<8192 x float> %vec)
  ret void
}

declare dso_local void @bar(<8192 x float>)
