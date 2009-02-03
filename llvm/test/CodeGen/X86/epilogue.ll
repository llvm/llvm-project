; RUN: llvm-as < %s | llc -march=x86 | not grep lea
; RUN: llvm-as < %s | llc -march=x86 | grep {movl	%ebp}

declare void @bar(<2 x i64>* %n)

define void @foo(i64 %h) {
  %k = trunc i64 %h to i32
  %p = alloca <2 x i64>, i32 %k
  call void @bar(<2 x i64>* %p)
  ret void
}
