; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 --amdgpu-memcpy-loop-unroll=100000 < %s | FileCheck --check-prefixes=GCN,GFX11 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 --amdgpu-memcpy-loop-unroll=100000 < %s | FileCheck --check-prefixes=GCN,GFX12 %s

;; Verify that inst_pref_size resolves to the correct value in the object file.
;; COMPUTE_PGM_RSRC3 is at offset 0x2C in each 64-byte kernel descriptor.
;; inst_pref_size is bits [9:4] on GFX11 (6-bit) and bits [11:4] on GFX12+ (8-bit).
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 --amdgpu-memcpy-loop-unroll=100000 -filetype=obj < %s -o %t.gfx11.o
; RUN: llvm-objdump -s -j .rodata %t.gfx11.o | FileCheck --check-prefix=OBJ-GFX11 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 --amdgpu-memcpy-loop-unroll=100000 -filetype=obj < %s -o %t.gfx12.o
; RUN: llvm-objdump -s -j .rodata %t.gfx12.o | FileCheck --check-prefix=OBJ-GFX12 %s

; The inst_pref_size is computed via MCExpr label subtraction, resolved at
; assembly/link time. In text output it appears as:
;   ((instprefsize(<code_size>)<<Shift)&Mask)>>Shift
; where:
;   <code_size>       = .Lfunc_endN - func_sym (exact function code size in bytes)
;   instprefsize      = min(divideCeil(code_size, cache_line_size), (1 << field_width) - 1)
;   field_width and cache_line_size are derived from the subtarget

; GCN-LABEL: .amdhsa_kernel large
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-large)<<4)&1008)>>4
; GFX11: codeLenInByte = {{[0-9]+}}
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-large)<<4)&4080)>>4
; GFX12: codeLenInByte = {{[0-9]+}}
;; Object: kernel descriptor at 0x00, COMPUTE_PGM_RSRC3 at 0x2C:
;; gfx11 pref=3 (0x30), gfx12 pref=4 (0x40)
; OBJ-GFX11: 0020 {{.*}}30000000
; OBJ-GFX12: 0020 {{.*}}40000000
define amdgpu_kernel void @large(ptr addrspace(1) %out, ptr addrspace(1) %in) {
bb:
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 256, i1 false)
  ret void
}

; GCN-LABEL: .amdhsa_kernel small
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-small)<<4)&1008)>>4
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-small)<<4)&4080)>>4
; GCN: codeLenInByte = {{[0-9]+}}
;; Object: kernel descriptor at 0x40, COMPUTE_PGM_RSRC3 at 0x6C:
;; pref=1 (0x10) for both
; OBJ-GFX11: 0060 {{.*}}10000000
; OBJ-GFX12: 0060 {{.*}}10000000
define amdgpu_kernel void @small() {
bb:
  ret void
}

; Inline asm is accounted for via MCExpr label subtraction (exact code size).
; The MCExpr resolves to the correct inst_pref_size at assembly time.

; GCN-LABEL: .amdhsa_kernel inline_asm
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-inline_asm)<<4)&1008)>>4
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-inline_asm)<<4)&4080)>>4
; GCN: codeLenInByte = {{[0-9]+}}
;; Object: kernel descriptor at 0x80, COMPUTE_PGM_RSRC3 at 0xAC:
;; pref=9 (0x90) for both
;; (.fill 256, 4, 0 = 1024 bytes + 4 s_endpgm = 1028 -> divideCeil(1028,128) = 9)
; OBJ-GFX11: 00a0 {{.*}}90000000
; OBJ-GFX12: 00a0 {{.*}}90000000
define amdgpu_kernel void @inline_asm() {
bb:
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}
