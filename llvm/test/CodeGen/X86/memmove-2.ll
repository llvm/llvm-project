; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-pc-linux-gnu | not grep call

declare void @llvm.memmove.i64(i8* %d, i8* %s, i64 %l, i32 %a)

define void @foo(i8* noalias %d, i8* noalias %s)
{
  call void @llvm.memmove.i64(i8* %d, i8* %s, i64 32, i32 1)
  ret void
}
