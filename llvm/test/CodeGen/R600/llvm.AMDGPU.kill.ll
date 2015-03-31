; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}kill_gs_const:
; SI-NOT: v_cmpx_le_f32
; SI: s_mov_b64 exec, 0

define void @kill_gs_const() #0 {
main_body:
  %0 = icmp ule i32 0, 3
  %1 = select i1 %0, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %1)
  %2 = icmp ule i32 3, 0
  %3 = select i1 %2, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %3)
  ret void
}

declare void @llvm.AMDGPU.kill(float)

attributes #0 = { "ShaderType"="2" }

!0 = !{!"const", null, i32 1}
