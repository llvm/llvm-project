; RUN: opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -S --passes='cgscc(inline)' < %s | FileCheck %s

; The internal callee contains a `noduplicate` intrinsic, so the inliner must
; not inline it at the two call sites below (doing so would duplicate the
; noduplicate call).

define internal void @barrier() {
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  ret void
}

define ptx_kernel void @noinline() {
start:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cond = icmp slt i32 %tid, 5
  br i1 %cond, label %bb1, label %bb2
bb1:
  ; CHECK: call void @barrier()
  call void @barrier()
  br label %exit
bb2:
  ; CHECK: call void @barrier()
  call void @barrier()
  br label %exit
exit:
  ret void
}
