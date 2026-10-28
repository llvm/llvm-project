; A module that already describes its kernels in !air.kernel keeps their
; signatures: the description names the parameters as they are.
; RUN: opt -mtriple=air -passes=air-scalar-args-to-constant-buffers -S %s | FileCheck %s

; CHECK: define void @kernel(float %alpha, ptr addrspace(1) %out)

define void @kernel(float %alpha, ptr addrspace(1) %out) {
  store float %alpha, ptr addrspace(1) %out
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel}
