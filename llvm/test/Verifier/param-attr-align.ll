; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: huge alignments are not supported yet
define dso_local void @foo(ptr %p) {
entry:
  call void @bar(ptr noundef byval(<8 x float>) align 8589934592 %p)
  ret void
}

declare dso_local void @bar(ptr %p)
