; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 --amdgpu-memcpy-loop-unroll=100000 < %s | FileCheck --check-prefixes=GCN,GFX11 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 --amdgpu-memcpy-loop-unroll=100000 < %s | FileCheck --check-prefixes=GCN,GFX12 %s

;; Verify that inst_pref_size resolves to the correct value in the object file.
;; COMPUTE_PGM_RSRC3 is at offset 0x2C in each 64-byte kernel descriptor.
;; GFX11 inst_pref_size is bits [9:4], so value N is encoded as N << 4.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 --amdgpu-memcpy-loop-unroll=100000 -filetype=obj < %s -o %t.o
; RUN: llvm-objdump -s -j .rodata %t.o | FileCheck --check-prefix=OBJ %s

; The inst_pref_size is computed via MCExpr label subtraction
; (code_end - func_sym), which resolves at assembly/link time.
; In text output it appears as a symbolic expression.

; GCN-LABEL: .amdhsa_kernel large
; GFX11: .amdhsa_inst_pref_size {{.*}}instprefsize({{.*}}large, 6){{.*}}
; GFX11: codeLenInByte = 3{{[0-9][0-9]$}}
; GFX12: .amdhsa_inst_pref_size {{.*}}instprefsize({{.*}}large, 8){{.*}}
; GFX12: codeLenInByte = 4{{[0-9][0-9]$}}
define amdgpu_kernel void @large(ptr addrspace(1) %out, ptr addrspace(1) %in) {
bb:
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 256, i1 false)
  ret void
}

; GCN-LABEL: .amdhsa_kernel small
; GCN: .amdhsa_inst_pref_size {{.*}}instprefsize({{.*}}small, {{[0-9]+}}){{.*}}
; GCN: codeLenInByte = {{[0-9]+$}}
define amdgpu_kernel void @small() {
bb:
  ret void
}

; Inline asm is accounted for via MCExpr label subtraction (exact code size).
; The MCExpr resolves to the correct inst_pref_size at assembly time.

; GCN-LABEL: .amdhsa_kernel inline_asm
; GCN: .amdhsa_inst_pref_size {{.*}}instprefsize({{.*}}inline_asm, {{[0-9]+}}){{.*}}
; GCN: codeLenInByte = {{[0-9]+$}}
define amdgpu_kernel void @inline_asm() {
bb:
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}

;; Object file checks: verify COMPUTE_PGM_RSRC3 at offset 0x2C in each KD.
;; COMPUTE_PGM_RSRC3 is the last dword on the 0x0020/0x0060/0x00a0 lines.
;; GFX11 inst_pref_size is bits [9:4], so value N is encoded as N << 4.
;;
;; large: 348 bytes -> pref_size=3 -> 3<<4=0x30
; OBJ: 0020 {{.*}}30000000
;; small: 4 bytes -> pref_size=1 -> 1<<4=0x10
; OBJ: 0060 {{.*}}10000000
;; inline_asm: 1028 bytes -> pref_size=9 -> 9<<4=0x90
; OBJ: 00a0 {{.*}}90000000
