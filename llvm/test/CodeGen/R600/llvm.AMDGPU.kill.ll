; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI %s

; SI-LABEL: @kill_gs
; SI: V_CMPX_LE_F32

define void @kill_gs() #0 {
main_body:
  %0 = icmp ule i32 0, 3
  %1 = select i1 %0, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %1)
  ret void
}

declare void @llvm.AMDGPU.kill(float)

attributes #0 = { "ShaderType"="2" }

!0 = metadata !{metadata !"const", null, i32 1}
