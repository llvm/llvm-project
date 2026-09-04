; RUN: llc -mtriple=amdgpu6.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -mtriple=amdgpu7.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -mtriple=amdgpu8.01 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgpu9.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgpu9.02 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgpu10.10 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgpu11.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s

; REQUIRES: asserts

; NOT-SUPPORTED: xnack setting for subtarget: Unsupported
; ANY: xnack setting for subtarget: Any
define void @xnack-subtarget-feature-any() #0 {
  ret void
}

attributes #0 = { nounwind }
