; RUN: llvm-as < %s | opt -mergefunc | llvm-dis | grep {alias weak} | count 2

define weak i32 @sum(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  ret i32 %sum
}

define weak i32 @add(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  ret i32 %sum
}
