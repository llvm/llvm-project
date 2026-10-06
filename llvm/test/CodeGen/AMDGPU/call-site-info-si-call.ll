; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -global-isel=0 --call-graph-section \
; RUN:   -stop-after=finalize-isel < %s \
; RUN:   | FileCheck --check-prefixes=CHECK,SDAG %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -global-isel=1 --call-graph-section \
; RUN:   -stop-after=finalize-isel < %s \
; RUN:   | FileCheck --check-prefixes=CHECK,GISEL %s

; The custom inserter replaces SI_CALL_ISEL with SI_CALL. The call site info
; has to move to the new instruction.
; GlobalISel selects SI_CALL directly and records no entry for this call.

; SDAG:       callSites:
; SDAG-NEXT:    - { bb: 0, offset: {{[0-9]+}}, fwdArgRegs: [] }
; GISEL:      callSites: []
; CHECK:      SI_CALL {{.*}}csr_amdgpu

define void @f(ptr inreg %fp) {
  call void %fp()
  ret void
}
