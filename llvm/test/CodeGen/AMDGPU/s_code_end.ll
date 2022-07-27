; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN,GCN-ASM,GFX10END-ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=obj < %s | llvm-objdump --arch=amdgcn --mcpu=gfx1010 -d - | FileCheck --check-prefixes=GCN,GCN-OBJ,GFX10END-OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1010 -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN,GCN-ASM,GFX10END-ASM %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN,GCN-ASM,GFX10NOEND %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1010 -filetype=obj < %s | llvm-objdump --arch=amdgcn --mcpu=gfx1010 -d - | FileCheck --check-prefixes=GCN,GCN-OBJ,GFX10NOEND,GFX10NOEND-OBJ %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN,GCN-ASM,GFX11END-ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj < %s | llvm-objdump --arch=amdgcn --mcpu=gfx1100 -d - | FileCheck --check-prefixes=GCN,GCN-OBJ,GFX11END-OBJ %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN,GCN-ASM,GFX90AEND-ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj < %s | llvm-objdump --arch=amdgcn --mcpu=gfx90a --disassemble - | FileCheck -check-prefixes=GCN,GCN-OBJ,GFX90AEND-OBJ %s

; GCN:            a_kernel1{{>?}}:
; GCN:                    s_endpgm
; GCN-ASM:        [[END_LABEL1:\.Lfunc_end.*]]:
; GCN-ASM-NEXT:           .size   a_kernel1, [[END_LABEL1]]-a_kernel1

; GCN-OBJ-NEXT:           s_nop 0

define amdgpu_kernel void @a_kernel1() #0 {
  ret void
}

; GCN:            a_kernel2{{>?}}:
; GCN:                    s_endpgm
; GCN-ASM:        [[END_LABEL2:\.Lfunc_end.*]]:
; GCN-ASM-NEXT:           .size   a_kernel2, [[END_LABEL2]]-a_kernel2

; GCN-OBJ:   {{^$}}

define amdgpu_kernel void @a_kernel2() #0 {
  ret void
}

; GCN-ASM:                .globl  a_function
; GCN-ASM-NEXT:           .p2align        2
; GCN-ASM-NEXT:           .type   a_function,@function

; GCN-NEXT:       a_function{{>?}}:
; GCN:                    s_setpc_b64
; GCN-ASM-NEXT:   [[END_LABEL3:\.Lfunc_end.*]]:
; GCN-ASM-NEXT:           .size   a_function, [[END_LABEL3]]-a_function
; GFX10END-ASM:           .p2alignl 6, 3214868480
; GFX11END-ASM:           .p2alignl 7, 3214868480
; GFX90AEND-ASM:          .p2alignl 6, 3212836864
; GFX10END-ASM-NEXT:      .fill 48, 4, 3214868480
; GFX11END-ASM-NEXT:      .fill 96, 4, 3214868480
; GFX90AEND-ASM-NEXT:     .fill 256, 4, 3212836864
; GFX10NOEND-NOT:         .fill
; GFX11NOEND-NOT:         .fill

; GFX10NOEND-OBJ-NOT:     s_code_end
; GFX10END-OBJ-NEXT:      s_code_end
; GFX11NOEND-OBJ-NOT:     s_code_end
; GFX11END-OBJ-NEXT:      s_code_end
; GFX90AEND-OBJ-NEXT:     s_nop 0

; GFX10END-OBJ:           s_code_end // 000000000140:
; GFX10END-OBJ-COUNT-47:  s_code_end
; GFX11END-OBJ:           s_code_end // 000000000140:
; GFX11END-OBJ-COUNT-47:  s_code_end
; GFX90AEND-OBJ:           s_nop 0 // 000000000140:
; GFX90AEND-OBJ-COUNT-255: s_nop 0

define void @a_function() #0 {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,512" }
