; RUN: llvm-as < %s | llc -march=x86-64 | grep -v cmp

define i64 @foo(i64 %x) {
  %t = icmp slt i64 %x, 1
  %r = zext i1 %t to i64
  ret i64 %r
}
