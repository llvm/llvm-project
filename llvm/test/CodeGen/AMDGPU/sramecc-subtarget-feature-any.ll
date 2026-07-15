; RUN: llc -mtriple=amdgpu7.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -mtriple=amdgpu9.06 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgpu9.08 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgpu12.50 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s

; REQUIRES: asserts

; NOT-SUPPORTED: sramecc setting for subtarget: Unsupported
; ANY: sramecc setting for subtarget: Any
define void @sramecc-subtarget-feature-default() #0 {
  ret void
}

attributes #0 = { nounwind }
