; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}kill_gs_const:
; SI-NOT: v_cmpx_le_f32
; SI: s_mov_b64 exec, 0

define amdgpu_gs void @kill_gs_const() {
main_body:
  %0 = icmp ule i32 0, 3
  %1 = select i1 %0, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %1)
  %2 = icmp ule i32 3, 0
  %3 = select i1 %2, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %3)
  ret void
}

; SI-LABEL: {{^}}kill_vcc_implicit_def:
; SI-NOT: v_cmp_gt_f32_e32 vcc,
; SI: v_cmp_gt_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], 0, v{{[0-9]+}}
; SI: v_cmpx_le_f32_e32 vcc, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
define amdgpu_ps void @kill_vcc_implicit_def([6 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <4 x i32>] addrspace(2)* byval, [34 x <8 x i32>] addrspace(2)* byval, float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, i32, float, float) {
entry:
  %tmp0 = fcmp olt float %13, 0.0
  call void @llvm.AMDGPU.kill(float %14)
  %tmp1 = select i1 %tmp0, float 1.0, float 0.0
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 1, i32 1, float %tmp1, float %tmp1, float %tmp1, float %tmp1)
  ret void
}

declare void @llvm.AMDGPU.kill(float)
declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

!0 = !{!"const", null, i32 1}
