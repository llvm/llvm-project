; RUN: llc -march=amdgcn < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=GCN,FUNC  %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=GCN,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=EG,FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=CM,FUNC %s

; FUNC-LABEL: {{^}}test:
; EG: LOG_IEEE
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; GCN: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
define void @test(ptr addrspace(1) %out, float %in) {
entry:
   %res = call float @llvm.log10.f32(float %in)
   store float %res, ptr addrspace(1) %out
   ret void
}

; FUNC-LABEL: {{^}}testv2:
; EG: LOG_IEEE
; EG: LOG_IEEE
; FIXME: We should be able to merge these packets together on Cayman so we
; have a maximum of 4 instructions.
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; GCN-DAG: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
define void @testv2(ptr addrspace(1) %out, <2 x float> %in) {
entry:
  %res = call <2 x float> @llvm.log10.v2f32(<2 x float> %in)
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}testv4:
; EG: LOG_IEEE
; EG: LOG_IEEE
; EG: LOG_IEEE
; EG: LOG_IEEE
; FIXME: We should be able to merge these packets together on Cayman so we
; have a maximum of 4 instructions.
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; CM-DAG: LOG_IEEE T{{[0-9]+\.[XYZW]}}
; GCN-DAG: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_log_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3e9a209a, v{{[0-9]+}}
define void @testv4(ptr addrspace(1) %out, <4 x float> %in) {
entry:
  %res = call <4 x float> @llvm.log10.v4f32(<4 x float> %in)
  store <4 x float> %res, ptr addrspace(1) %out
  ret void
}

declare float @llvm.log10.f32(float) readnone
declare <2 x float> @llvm.log10.v2f32(<2 x float>) readnone
declare <4 x float> @llvm.log10.v4f32(<4 x float>) readnone
