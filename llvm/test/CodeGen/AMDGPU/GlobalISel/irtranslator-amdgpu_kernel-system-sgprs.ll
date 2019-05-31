; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -O0 -amdgpu-ir-lower-kernel-arguments=0 -stop-after=irtranslator -global-isel %s -o - | FileCheck -check-prefix=HSA %s

; HSA-LABEL: name: default_kernel
; HSA: liveins:
; HSA-NEXT: - { reg: '$sgpr0_sgpr1_sgpr2_sgpr3', virtual-reg: '%0' }
; HSA-NEXT: - { reg: '$sgpr4', virtual-reg: '%1' }
; HSA-NEXT: frameInfo:
define amdgpu_kernel void @default_kernel() {
  ret void
}
