; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i64 0} | count 2

define i64 @foo(i32 %x) nounwind {
  %y = lshr i32 %x, 1
  %r = udiv i32 %y, -1
  %z = sext i32 %r to i64
  ret i64 %z
}
define i64 @bar(i32 %x) nounwind {
  %y = lshr i32 %x, 31
  %r = udiv i32 %y, 3
  %z = sext i32 %r to i64
  ret i64 %z
}
