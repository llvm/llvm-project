; RUN: llc -mtriple=aarch64-- < %s

; This regression test is defending against using the wrong interface for TypeSize.
; This issue appeared in DAGCombiner::visitLIFETIME_END when visiting a LIFETIME_END
; node linked to a scalable store.

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define void @foo(<vscale x 4 x i32>* nocapture dereferenceable(16) %ptr) {
entry:
  %tmp = alloca <vscale x 4 x i32>, align 8
  %tmp_ptr = bitcast <vscale x 4 x i32>* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %tmp_ptr)
  store <vscale x 4 x i32> undef, <vscale x 4 x i32>* %ptr
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %tmp_ptr)
  ret void
}
