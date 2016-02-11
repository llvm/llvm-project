; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() #1

; SI-LABEL: {{^}}v_cnd_nan_nosgpr:
; SI: v_cndmask_b32_e64 v{{[0-9]}}, v{{[0-9]}}, -1, s{{\[[0-9]+:[0-9]+\]}}
; SI-DAG: v{{[0-9]}}
; All nan values are converted to 0xffffffff
; SI: s_endpgm
define void @v_cnd_nan_nosgpr(float addrspace(1)* %out, i32 %c, float addrspace(1)* %fptr) #0 {
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %f.gep = getelementptr float, float addrspace(1)* %fptr, i32 %idx
  %f = load float, float addrspace(1)* %fptr
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, float 0xFFFFFFFFE0000000, float %f
  store float %select, float addrspace(1)* %out
  ret void
}


; This requires slightly trickier SGPR operand legalization since the
; single constant bus SGPR usage is the last operand, and it should
; never be moved.

; SI-LABEL: {{^}}v_cnd_nan:
; SI: v_cndmask_b32_e64 v{{[0-9]}}, v{{[0-9]}}, -1, s{{\[[0-9]+:[0-9]+\]}}
; SI-DAG: v{{[0-9]}}
; All nan values are converted to 0xffffffff
; SI: s_endpgm
define void @v_cnd_nan(float addrspace(1)* %out, i32 %c, float %f) #0 {
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, float 0xFFFFFFFFE0000000, float %f
  store float %select, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
