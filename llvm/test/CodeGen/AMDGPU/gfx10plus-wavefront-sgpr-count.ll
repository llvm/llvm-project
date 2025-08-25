; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s

; CHECK-NOT: {{[Ee]rror.*COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT}}
define amdgpu_kernel void @test(i128 inreg) { 
  ret void 
}
