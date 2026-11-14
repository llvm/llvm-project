; RUN: not llvm-as %s -disable-output 2>/dev/null

define void @foo(ptr %p) {
  store i1 false, ptr %p, align 8589934592
  ret void
}
