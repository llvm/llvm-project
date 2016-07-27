; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}rcp_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define void @rcp_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv float 1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define void @rcp_ulp25_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_fast_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define void @rcp_fast_ulp25_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv fast float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_arcp_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define void @rcp_arcp_ulp25_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv arcp float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_global_fast_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define void @rcp_global_fast_ulp25_pat_f32(float addrspace(1)* %out, float %src) #2 {
  %rcp = fdiv float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_fabs_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], |[[SRC]]|
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define void @rcp_fabs_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %src.fabs = call float @llvm.fabs.f32(float %src)
  %rcp = fdiv float 1.0, %src.fabs
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FIXME: fneg folded into constant 1
; FUNC-LABEL: {{^}}rcp_fabs_fneg_pat_f32:
define void @rcp_fabs_fneg_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %src.fabs = call float @llvm.fabs.f32(float %src)
  %src.fabs.fneg = fsub float -0.0, %src.fabs
  %rcp = fdiv float 1.0, %src.fabs.fneg
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}


declare float @llvm.fabs.f32(float) #1

attributes #0 = { nounwind "unsafe-fp-math"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "unsafe-fp-math"="true" }

!0 = !{float 2.500000e+00}
