; XFAIL: *
; REQUIRES: asserts
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s

; write_register doesn't prevent us from illegally trying to write a
; vgpr value into a scalar register, but I don't think there's much we
; can do to avoid this.

declare void @llvm.write_register.i32(metadata, i32) #0
declare i32 @llvm.amdgcn.workitem.id.x() #0
declare void @llvm.amdgcn.wave.barrier() #2

define amdgpu_kernel void @write_vgpr_into_sgpr() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.write_register.i32(metadata !0, i32 %tid)
  call void @llvm.amdgcn.wave.barrier() #2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { convergent nounwind }

!0 = !{!"exec_lo"}
