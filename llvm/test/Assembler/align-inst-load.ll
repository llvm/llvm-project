; RUN: not llvm-as %s -o /dev/null 2>/dev/null

define void @foo() {
  load i1* %p, align 1073741824
  ret void
}
