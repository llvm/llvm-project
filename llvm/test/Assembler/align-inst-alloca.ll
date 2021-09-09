; RUN: not llvm-as %s -o /dev/null 2>/dev/null

define void @foo() {
  %p = alloca i1, align 2147483648
  ret void
}
