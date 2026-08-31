; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --syms - | FileCheck %s --check-prefix=ELF
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; Test that dynamic LDS used by a device function (not a kernel) works with
; object linking. The device function's use generates a relocation, and the
; AMDGPU info section records the (function, dyn_var) association.

@dyn_lds = external addrspace(3) global [0 x i8], align 8

declare void @extern_func()

define void @device_func(i32 %idx) {
  %gep = getelementptr [0 x i8], ptr addrspace(3) @dyn_lds, i32 0, i32 %idx
  store i8 99, ptr addrspace(3) %gep
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @my_kernel(i32 %idx) {
  call void @device_func(i32 %idx)
  ret void
}

; ELF:      R_AMDGPU_ABS32_LO dyn_lds

; ELF:      Symbol {
; ELF:        Name: dyn_lds
; ELF-NEXT:   Value: 0x8
; ELF-NEXT:   Size: 0
; ELF-NEXT:   Binding: Global
; ELF-NEXT:   Type: Object
; ELF-NEXT:   Other: 0
; ELF-NEXT:   Section: Processor Specific (0xFF00)
; ELF-NEXT: }

; ASM-DAG: .amdgpu_lds dyn_lds, 0, 8
; ASM-DAG: .amdgpu_use dyn_lds
