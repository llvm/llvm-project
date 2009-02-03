; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-apple-darwin10 | grep __bzero

declare void @llvm.memset.i64(i8*, i8, i64, i32)

define void @foo(i8* %p, i64 %n) {
  call void @llvm.memset.i64(i8* %p, i8 0, i64 %n, i32 4)
  ret void
}
