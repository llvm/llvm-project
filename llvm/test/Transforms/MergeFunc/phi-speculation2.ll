; RUN: llvm-as < %s | opt -mergefunc -stats -disable-output |& grep {functions merged}

define i32 @foo1(i32 %x) {
entry:
  %A = add i32 %x, 1
  %B = call i32 @foo1(i32 %A)
  br label %loop
loop:
  %C = phi i32 [%B, %entry], [%D, %loop]
  %D = add i32 %C, 2
  %E = icmp ugt i32 %D, 10000
  br i1 %E, label %loopexit, label %loop
loopexit:
  ret i32 %D
}

define i32 @foo2(i32 %x) {
entry:
  %0 = add i32 %x, 1
  %1 = call i32 @foo2(i32 %0)
  br label %loop
loop:
  %2 = phi i32 [%1, %entry], [%3, %loop]
  %3 = add i32 %2, 2
  %4 = icmp ugt i32 %3, 10000
  br i1 %4, label %loopexit, label %loop
loopexit:
  ret i32 %3
}
