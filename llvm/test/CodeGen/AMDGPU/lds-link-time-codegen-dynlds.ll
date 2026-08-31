; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --syms - | FileCheck %s --check-prefix=ELF
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; Test that dynamic LDS (extern __shared__, represented as zero-sized
; addrspace(3) globals) works with object linking. Dynamic LDS variables
; flow through the Global-scope path -- no per-kernel representative is
; created. The original symbol name is preserved. Uses are recorded in the
; unified .amdgpu.info section via .amdgpu_use.

@static_lds = addrspace(3) global [64 x float] poison, align 16
@dyn_lds = external addrspace(3) global [0 x i8], align 4

declare void @extern_func()

define void @helper(i32 %idx) {
  %gep = getelementptr [64 x float], ptr addrspace(3) @static_lds, i32 0, i32 %idx
  store float 2.0, ptr addrspace(3) %gep
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @test_kernel(i32 %idx) {
  %gep.static = getelementptr [64 x float], ptr addrspace(3) @static_lds, i32 0, i32 %idx
  store float 1.0, ptr addrspace(3) %gep.static
  %gep.dyn = getelementptr [0 x i8], ptr addrspace(3) @dyn_lds, i32 0, i32 %idx
  store i8 42, ptr addrspace(3) %gep.dyn
  call void @helper(i32 %idx)
  ret void
}

; Relocations appear first in llvm-readobj output.
; ELF:      R_AMDGPU_ABS32_LO dyn_lds

; The resources section should reference the dyn_lds symbol.
; ELF:      R_AMDGPU_ABS64 dyn_lds

; The original dyn_lds symbol should be emitted as size-0 SHN_AMDGPU_LDS.
; ELF:      Symbol {
; ELF:        Name: dyn_lds
; ELF-NEXT:   Value: 0x4
; ELF-NEXT:   Size: 0
; ELF-NEXT:   Binding: Global
; ELF-NEXT:   Type: Object
; ELF-NEXT:   Other: 0
; ELF-NEXT:   Section: Processor Specific (0xFF00)
; ELF-NEXT: }

; ASM-DAG: .amdgpu_lds dyn_lds, 0, 4
; ASM-DAG: .amdgpu_use dyn_lds
