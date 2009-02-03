; RUN: llvm-as < %s | llc -march=x86-64 | grep -v cmp

define i64 @foo(i64 %x) {
  %t = icmp eq i64 %x, 0
  %r = zext i1 %t to i64
  ret i64 %r
}
