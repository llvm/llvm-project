; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare float @llvm.amdgcn.cubema(float, float, float) #0

; GCN-LABEL: {{^}}test_cubema:
; GCN: v_cubema_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @test_cubema(float addrspace(1)* %out, float %a, float %b, float %c) #1 {
  %result = call float @llvm.amdgcn.cubema(float %a, float %b, float %c)
  store float %result, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
