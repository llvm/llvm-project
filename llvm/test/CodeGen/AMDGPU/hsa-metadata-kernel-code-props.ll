; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX800 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]

; CHECK:  Kernels:
; CHECK:    - Name: test
; CHECK:      CodeProps:
; CHECK:        KernargSegmentSize:  24
; GFX700:       WavefrontNumSGPRs:   6
; GFX800:       WavefrontNumSGPRs:   96
; GFX900:       WavefrontNumSGPRs:   6
; GFX700:       WorkitemNumVGPRs:    4
; GFX800:       WorkitemNumVGPRs:    6
; GFX900:       WorkitemNumVGPRs:    6
; CHECK:        KernargSegmentAlign: 4
; CHECK:        GroupSegmentAlign:   4
; CHECK:        PrivateSegmentAlign: 4
; CHECK:        WavefrontSize:       6
define amdgpu_kernel void @test(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}
