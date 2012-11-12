; RUN: opt  %s -dse -disable-output
; test that we don't crash
declare void @bar()

define void @foo() {
bb1:
  %memtmp3.i = alloca [21 x i8], align 1
  %0 = getelementptr inbounds [21 x i8]* %memtmp3.i, i64 0, i64 0
  br label %bb3

bb2:
  call void @llvm.lifetime.end(i64 -1, i8* %0)
  br label %bb3

bb3:
  call void @bar()
  call void @llvm.lifetime.end(i64 -1, i8* %0)
  br label %bb4

bb4:
  ret void

}

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
