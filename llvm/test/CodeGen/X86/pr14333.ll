; RUN: llc -mtriple=x86_64-unknown-unknown < %s
%foo = type { i64, i64 }
define void @bar(ptr %zed) {
  store i64 0, ptr %zed, align 8
  %tmp2 = getelementptr inbounds %foo, ptr %zed, i64 0, i32 1
  store i64 0, ptr %tmp2, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %zed, i8 0, i64 16, i1 false)
  ret void
}
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
