; RUN: llc < %s -mtriple=i386-apple-darwin10 | grep __bzero

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind

define void @foo(i8* %p, i32 %len) {
  call void @llvm.memset.p0i8.i32(i8* %p, i8 0, i32 %len, i1 false)
  ret void
}
