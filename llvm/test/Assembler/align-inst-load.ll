; RUN: not llvm-as %s -disable-output 2>/dev/null

define void @foo(ptr %p) {
  load i1, ptr %p, align 8589934592
  ret void
}
