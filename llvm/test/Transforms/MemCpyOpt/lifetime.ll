; RUN: opt < %s -O1 -S | FileCheck %s

; performCallSlotOptzn in MemCpy should not exchange the calls to
; @llvm.lifetime.start and @llvm.memcpy.

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1
declare void @llvm.lifetime.start(i64, i8* nocapture) #1
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

define void @_ZN4CordC2EOS_(i8* nocapture dereferenceable(16) %arg1) {
bb:
; CHECK-LABEL: @_ZN4CordC2EOS_
; CHECK-NOT: call void @llvm.lifetime.start
; CHECK: ret void
  %tmp = alloca [8 x i8], align 8
  %tmp5 = bitcast [8 x i8]* %tmp to i8*
  call void @llvm.lifetime.start(i64 16, i8* %tmp5)
  %tmp10 = getelementptr inbounds i8, i8* %tmp5, i64 7
  store i8 0, i8* %tmp10, align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %arg1, i8* %tmp5, i64 16, i32 8, i1 false)
  call void @llvm.lifetime.end(i64 16, i8* %tmp5)
  ret void
}

attributes #1 = { argmemonly nounwind }
