; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-pc-linux-gnu | grep {call	memmove}

declare void @llvm.memmove.i64(i8* %d, i8* %s, i64 %l, i32 %a)

define void @foo(i8* %d, i8* %s)
{
  call void @llvm.memmove.i64(i8* %d, i8* %s, i64 32, i32 1)
  ret void
}
