; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa --call-graph-section \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s

; The custom inserter replaces SI_CALL_ISEL with SI_CALL. The call site info
; has to move to the new instruction.

; CHECK:      callSites:
; CHECK-NEXT:   - { bb: 0, offset: {{[0-9]+}}, fwdArgRegs: [] }
; CHECK:      SI_CALL killed

define void @f(ptr inreg %fp) {
  call void %fp()
  ret void
}
