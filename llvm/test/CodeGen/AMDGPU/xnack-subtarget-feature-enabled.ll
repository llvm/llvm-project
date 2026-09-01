; RUN: llc -mtriple=amdgpu6.00 -debug-only=gcn-subtarget -filetype=null %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -mtriple=amdgpu7.00 -debug-only=gcn-subtarget -filetype=null %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -mtriple=amdgpu8.01 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu9.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu9.06 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu10.10 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu11.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=WARN %s

; REQUIRES: asserts

; WARN: warning: xnack 'On' was requested for a processor that does not support it!
; ON: xnack setting for subtarget: On
define void @xnack-subtarget-feature-enabled() #0 {
  ret void
}

attributes #0 = { "target-features"="+xnack" }
