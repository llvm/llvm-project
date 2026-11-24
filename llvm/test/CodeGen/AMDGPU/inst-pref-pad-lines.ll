; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa < %s | FileCheck --check-prefixes=NOPAD,NOPAD-GFX12 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa < %s | FileCheck --check-prefixes=NOPAD,NOPAD-GFX11 %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -amdgpu-inst-pref-pad-lines=2 < %s | FileCheck --check-prefixes=PAD2,PAD2-GFX12 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -amdgpu-inst-pref-pad-lines=2 < %s | FileCheck --check-prefixes=PAD2,PAD2-GFX11 %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-inst-pref-pad-lines=2 < %s | FileCheck --check-prefixes=PAD2,PAD2-GFX12 %s

;; Resolved kernel descriptors. COMPUTE_PGM_RSRC3 is at +0x2C in each
;; 64-byte descriptor. inst_pref_size is bits [11:4] (GFX12+) or [9:4] (GFX11).
;; An empty kernel occupies 1 cache line. Cache lines are 128 bytes.
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -amdgpu-inst-pref-pad-lines=2 -filetype=obj < %s -o %t.pad2.gfx12.o
; RUN: llvm-objdump -s -j .rodata %t.pad2.gfx12.o | FileCheck --check-prefix=OBJ-PAD2 %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -amdgpu-inst-pref-pad-lines=2 < %s | llvm-mc -triple amdgpu12.00-amd-amdhsa -mcpu=gfx1200 -filetype=obj -o %t.rt.o
; RUN: llvm-objdump -s -j .rodata %t.rt.o | FileCheck --check-prefix=OBJ-PAD2 %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -amdgpu-inst-pref-pad-lines=1000 -filetype=obj < %s -o %t.clamp.gfx12.o
; RUN: llvm-objdump -s -j .rodata %t.clamp.gfx12.o | FileCheck --check-prefix=OBJ-CLAMP12 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -amdgpu-inst-pref-pad-lines=1000 -filetype=obj < %s -o %t.clamp.gfx11.o
; RUN: llvm-objdump -s -j .rodata %t.clamp.gfx11.o | FileCheck --check-prefix=OBJ-CLAMP11 %s

; The global option adds 2 lines (256 bytes) to the code-size expression.
; The kernel body stays a single s_endpgm.
; PAD2-LABEL: small:
; PAD2: s_endpgm

; NOPAD-LABEL: .amdhsa_kernel small
; NOPAD-GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-small)<<4)&4080)>>4
; NOPAD-GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-small)<<4)&1008)>>4
; PAD2-LABEL: .amdhsa_kernel small
; PAD2-GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-small+256)<<4)&4080)>>4
; PAD2-GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-small+256)<<4)&1008)>>4
;; pref = min(1 + 2, max) = 3 → 3 << 4 = 0x30
; OBJ-PAD2: 0020 {{.*}}30000000
;; pref = min(1 + 1000, 255) = 255 → 255 << 4 = 0xff0
; OBJ-CLAMP12: 0020 {{.*}}f00f0000
;; pref = min(1 + 1000, 63) = 63 → 63 << 4 = 0x3f0
; OBJ-CLAMP11: 0020 {{.*}}f0030000
define amdgpu_kernel void @small() {
bb:
  ret void
}

; A kernel attribute replaces the global option. 4 lines = 512 bytes.
; NOPAD-LABEL: .amdhsa_kernel attr_pad
; NOPAD-GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attr_pad+512)<<4)&4080)>>4
; NOPAD-GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attr_pad+512)<<4)&1008)>>4
; PAD2-LABEL: .amdhsa_kernel attr_pad
; PAD2-GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attr_pad+512)<<4)&4080)>>4
; PAD2-GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-attr_pad+512)<<4)&1008)>>4
;; pref = 1 + 4 = 5 → 5 << 4 = 0x50. Descriptor at 0x40, field at 0x6c.
; OBJ-PAD2: 0060 {{.*}}50000000
; OBJ-CLAMP12: 0060 {{.*}}50000000
; OBJ-CLAMP11: 0060 {{.*}}50000000
define amdgpu_kernel void @attr_pad() #0 {
bb:
  ret void
}

; An explicit 0 keeps today's unpadded size even when the global option is set.
; NOPAD-LABEL: .amdhsa_kernel attr_zero
; NOPAD-GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attr_zero)<<4)&4080)>>4
; NOPAD-GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attr_zero)<<4)&1008)>>4
; PAD2-LABEL: .amdhsa_kernel attr_zero
; PAD2-GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attr_zero)<<4)&4080)>>4
; PAD2-GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-attr_zero)<<4)&1008)>>4
;; pref = 1 → 0x10. Descriptor at 0x80, field at 0xac.
; OBJ-PAD2: 00a0 {{.*}}10000000
; OBJ-CLAMP12: 00a0 {{.*}}10000000
; OBJ-CLAMP11: 00a0 {{.*}}10000000
define amdgpu_kernel void @attr_zero() #1 {
bb:
  ret void
}

attributes #0 = { "amdgpu-inst-pref-pad-lines"="4" }
attributes #1 = { "amdgpu-inst-pref-pad-lines"="0" }
