; RUN: llc -march=amdgcn -mcpu=gfx1010 -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN,GCN-ASM,GFX10,GFX10-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -filetype=obj < %s | llvm-objdump -arch=amdgcn -mcpu=gfx1010 -disassemble - | FileCheck -check-prefixes=GCN,GCN-OBJ,GFX10,GFX10-OBJ %s

; GCN:            a_kernel1:
; GCN-NEXT:               s_endpgm
; GCN-ASM-NEXT:   [[END_LABEL1:\.Lfunc_end.*]]:
; GCN-ASM-NEXT:           .size   a_kernel1, [[END_LABEL1]]-a_kernel1
; GCN-ASM:                .section        .AMDGPU.config

; GCN-OBJ-NEXT:           s_nop 0

define amdgpu_kernel void @a_kernel1() {
  ret void
}

; GCN:            a_kernel2:
; GCN-NEXT:               s_endpgm
; GCN-ASM-NEXT:   [[END_LABEL2:\.Lfunc_end.*]]:
; GCN-ASM-NEXT:           .size   a_kernel2, [[END_LABEL2]]-a_kernel2
; GCN-ASM:                .section        .AMDGPU.config

; GCN-OBJ-NEXT:   {{^$}}

define amdgpu_kernel void @a_kernel2() {
  ret void
}

; GCN-ASM:                .text
; GCN-ASM-NEXT:           .globl  a_function
; GCN-ASM-NEXT:           .p2align        2
; GCN-ASM-NEXT:           .type   a_function,@function

; GCN-NEXT:       a_function:
; GCN:                    s_setpc_b64
; GCN-ASM-NEXT:   [[END_LABEL3:\.Lfunc_end.*]]:
; GCN-ASM-NEXT:           .size   a_function, [[END_LABEL3]]-a_function
; GFX10-ASM:              .p2alignl 6, 3214868480
; GFX10-ASM-NEXT:         .fill 32, 4, 3214868480

; GFX10-OBJ-NEXT:         s_code_end

; GFX10-OBJ:              s_code_end // 000000000140:
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end

; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end

; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end

; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end
; GFX10-OBJ-NEXT:         s_code_end

define void @a_function() {
  ret void
}
