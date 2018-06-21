; RUN: llc < %s -mtriple=i686-apple-darwin10 | grep __bzero

define void @foo(i8* %p, i64 %n) {
  call void @llvm.memset.p0i8.i64(i8* %p, i8 0, i64 %n, i32 4, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
