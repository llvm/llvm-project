; XFAIL: *
; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s

; write_register doesn't prevent us from illegally trying to write a
; vgpr value into a scalar register, but I don't think there's much we
; can do to avoid this.

declare void @llvm.write_register.i32(metadata, i32) nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare void @llvm.amdgcn.wave.barrier() convergent nounwind

define amdgpu_kernel void @write_vgpr_into_sgpr() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.write_register.i32(metadata !0, i32 %tid)
  call void @llvm.amdgcn.wave.barrier() convergent nounwind
  ret void
}

!0 = !{!"exec_lo"}
