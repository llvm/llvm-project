; RUN: not llvm-as %s -o /dev/null 2>/dev/null

define void @foo(i1* %p) {
  load i1, i1* %p, align 2147483648
  ret void
}
