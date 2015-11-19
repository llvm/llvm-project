; RUN: llc -mtriple=x86_64-unknown-unknown < %s
%foo = type { i64, i64 }
define void @bar(%foo* %zed) {
  %tmp = getelementptr inbounds %foo, %foo* %zed, i64 0, i32 0
  store i64 0, i64* %tmp, align 8
  %tmp2 = getelementptr inbounds %foo, %foo* %zed, i64 0, i32 1
  store i64 0, i64* %tmp2, align 8
  %tmp3 = bitcast %foo* %zed to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp3, i8 0, i64 16, i1 false)
  ret void
}
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
