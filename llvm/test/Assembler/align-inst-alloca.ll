; RUN: not llvm-as %s -disable-output 2>/dev/null

define void @foo() {
  %p = alloca i1, align 8589934592
  ret void
}
