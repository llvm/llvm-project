; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG-SAFE -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: Evergreen only ever does unsafe fp math.
; FUNC-LABEL: {{^}}rcp_pat_f32:
; EG: RECIP_IEEE
define void @rcp_pat_f32(float addrspace(1)* %out, float %src) nounwind {
  %rcp = fdiv float 1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}
