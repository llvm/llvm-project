; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s
; RUN: llc -mtriple=amdgpu10.31-amd-amdhsa -filetype=obj < %s 2>&1 | llvm-objdump -d --section=.rodata - | FileCheck %s

; CHECK-NOT: error
define amdgpu_kernel void @test(i128 inreg) { 
  ret void 
}
