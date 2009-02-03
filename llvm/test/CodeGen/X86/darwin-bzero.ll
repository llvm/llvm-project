; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin10 | grep __bzero

declare void @llvm.memset.i32(i8*, i8, i32, i32)

define void @foo(i8* %p, i32 %len) {
  call void @llvm.memset.i32(i8* %p, i8 0, i32 %len, i32 1);
  ret void
}
