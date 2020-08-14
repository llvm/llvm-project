; RUN: llc -march=amdgcn -mcpu=gfx600 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -march=amdgcn -mcpu=gfx802 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -march=amdgcn -mcpu=gfx902 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s

; REQUIRES: asserts

; Some subtargets have a default setting of 'On' instead of 'Any' to maintain
; backwards compatibility. This is a temporary measure until the new TargetID is
; implemented.

; NOT-SUPPORTED: XNACK setting for subtarget: Not Supported
; ANY: XNACK setting for subtarget: Any
; ON: XNACK setting for subtarget: On
define void @xnack-subtarget-feature-any() #0 {
  ret void
}

attributes #0 = { nounwind }
