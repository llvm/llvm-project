; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}kill_gs_const:
; SI-NOT: v_cmpx_le_f32
; SI: s_mov_b64 exec, 0
define amdgpu_gs void @kill_gs_const() {
main_body:
  %tmp = icmp ule i32 0, 3
  %tmp1 = select i1 %tmp, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %tmp1)
  %tmp2 = icmp ule i32 3, 0
  %tmp3 = select i1 %tmp2, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %tmp3)
  ret void
}

; SI-LABEL: {{^}}kill_vcc_implicit_def:
; SI-NOT: v_cmp_gt_f32_e32 vcc,
; SI: v_cmp_gt_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], 0, v{{[0-9]+}}
; SI: v_cmpx_le_f32_e32 vcc, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
define amdgpu_ps void @kill_vcc_implicit_def([6 x <16 x i8>] addrspace(2)* byval %arg, [17 x <16 x i8>] addrspace(2)* byval %arg1, [17 x <4 x i32>] addrspace(2)* byval %arg2, [34 x <8 x i32>] addrspace(2)* byval %arg3, float inreg %arg4, i32 inreg %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <3 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, i32 %arg19, float %arg20, float %arg21) {
entry:
  %tmp0 = fcmp olt float %arg13, 0.000000e+00
  call void @llvm.AMDGPU.kill(float %arg14)
  %tmp1 = select i1 %tmp0, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  ret void
}

declare void @llvm.AMDGPU.kill(float) #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0

attributes #0 = { nounwind }
