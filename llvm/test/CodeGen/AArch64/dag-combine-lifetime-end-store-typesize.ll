; RUN: llc -mtriple=aarch64-- < %s 2>&1 | FileCheck --allow-empty %s

; This regression test is defending against a TypeSize warning 'assumption that TypeSize is not
; scalable'. This warning appeared in DAGCombiner::visitLIFETIME_END when visiting a LIFETIME_END
; node linked to a scalable store.

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; CHECK-NOT: warning:

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
