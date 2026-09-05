; RUN: llc -O0 -global-isel -mtriple=amdgpu9.50-amd-amdhsa -verify-machineinstrs -filetype=null %s
; RUN: llc -O2 -global-isel -mtriple=amdgpu9.50-amd-amdhsa -verify-machineinstrs -filetype=null %s
; RUN: llc -O2 -global-isel -mtriple=amdgpu9.0a-amd-amdhsa -verify-machineinstrs -filetype=null %s

define amdgpu_kernel void @and_vgpr_s32(i32 %lane.bit) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lane.bit1 = and i32 %tid, 1
  %cond = icmp eq i32 %lane.bit1, 0
  br i1 %cond, label %common.ret, label %else

common.ret:
  ret void

else:
  %ins = insertelement <4 x i32> zeroinitializer, i32 %lane.bit, i32 0
  %ins1 = insertelement <4 x i32> %ins, i32 0, i32 0
  %ins2 = insertelement <4 x i32> %ins1, i32 0, i32 0
  %ins3 = insertelement <4 x i32> %ins2, i32 0, i32 0
  br label %common.ret
}

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
