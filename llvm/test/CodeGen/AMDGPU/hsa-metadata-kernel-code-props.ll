; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX800 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]

; CHECK:  Kernels:
; CHECK:    - Name:       test
; CHECK:      SymbolName: 'test@kd'
; CHECK:      CodeProps:
; CHECK:        KernargSegmentSize:      24
; CHECK:        GroupSegmentFixedSize:   0
; CHECK:        PrivateSegmentFixedSize: 0
; CHECK:        KernargSegmentAlign:     8
; CHECK:        WavefrontSize:           64
; GFX700:       NumSGPRs:                6
; GFX800:       NumSGPRs:                96
; GFX900:       NumSGPRs:                6
; GFX700:       NumVGPRs:                4
; GFX800:       NumVGPRs:                6
; GFX900:       NumVGPRs:                6
; CHECK:        MaxFlatWorkgroupSize:    256
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
