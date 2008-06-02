; RUN: llvm-as < %s | llc -march=c | grep {llvm_cbe_t.*&1}
define i32 @test(i32 %r) {
  %s = icmp eq i32 %r, 0
  %t = add i1 %s, %s
  %u = zext i1 %t to i32
  br i1 %t, label %A, label %B
A:

  ret i32 %u
B:

  %v = select i1 %t, i32 %r, i32 %u
  ret i32 %v
}
