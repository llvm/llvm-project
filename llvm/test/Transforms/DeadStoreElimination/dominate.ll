; RUN: opt -passes=dse -disable-output < %s
; test that we don't crash
declare void @bar()

define void @foo() {
bb1:
  %memtmp3.i = alloca [21 x i8], align 1
  br label %bb3

bb2:
  call void @llvm.lifetime.end.p0(ptr %memtmp3.i)
  br label %bb3

bb3:
  call void @bar()
  call void @llvm.lifetime.end.p0(ptr %memtmp3.i)
  br label %bb4

bb4:
  ret void

}

declare void @llvm.lifetime.end.p0(ptr nocapture) nounwind
