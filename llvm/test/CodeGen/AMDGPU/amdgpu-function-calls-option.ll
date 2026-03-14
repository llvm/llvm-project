; RUN: llc -mtriple=amdgcn-amd-amdhsa -amdgpu-function-calls < %s | FileCheck -check-prefix=CALLS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=CALLS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -amdgpu-function-calls=0 < %s | FileCheck -check-prefix=NOCALLS %s
; RUN: llc -mtriple=r600-mesa-mesa3d < %s | FileCheck -check-prefix=NOCALLS %s
; RUN: llc -mtriple=r600-mesa-mesa3d -amdgpu-function-calls=0 < %s | FileCheck -check-prefix=NOCALLS %s

; CALLS-LABEL: callee:
; CALLS: ;;#ASMSTART
; CALLS: ;;#ASMEND

; NOCALLS-NOT: callee
; R600-NOT: callee
define internal void @callee() {
  call void asm sideeffect "", ""()
  ret void
}

; CALLS-LABEL: kernel:
; CALLS: s_swappc_b64

; NOCALLS-LABEL: kernel:
; NOCALLS: ;;#ASMSTART
; NOCALLS: ;;#ASMEND
define amdgpu_kernel void @kernel() {
  call void @callee()
  ret void
}
