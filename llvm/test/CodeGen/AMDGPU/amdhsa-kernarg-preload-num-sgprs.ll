; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj < %s | llvm-objdump -s -j .rodata - | FileCheck --check-prefix=OBJDUMP %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck --check-prefix=ASM %s

; OBJDUMP: Contents of section .rodata:
; OBJDUMP-NEXT: 0000 00000000 00000000 10010000 00000000  ................
; OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NOT:  0030 0000af00 94130000 1a000400 00000000  ................
; OBJDUMP-NEXT: 0030 8000af00 98130000 1e000400 00000000  ................

; ASM-LABEL: amdhsa_kernarg_preload_4_implicit_6:
; ASM: .amdhsa_user_sgpr_count 12
; ASM: .amdhsa_next_free_sgpr 12
; ASM: ; TotalNumSgprs: 18
; ASM: ; NumSGPRsForWavesPerEU: 18

; Test that we include preloaded SGPRs in the GRANULATED_WAVEFRONT_SGPR_COUNT
; feild that are not explicitly referenced in the kernel. This test has 6 implicit
; user SPGRs enabled, 4 preloaded kernarg SGPRs, plus 6 extra SGPRs allocated
; for flat scratch, ect. The total number of allocated SGPRs encoded in the
; kernel descriptor should be 16. That's a 1 in the KD field since the granule
; size is 8 and it's NumGranules - 1. The encoding for that looks like '40'.

define amdgpu_kernel void @amdhsa_kernarg_preload_4_implicit_6(i128 inreg) { ret void }

; OBJDUMP-NEXT: 0040 00000000 00000000 20010000 00000000  ........ .......
; OBJDUMP-NEXT: 0050 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 0060 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 0070 4000af00 94000000 08000800 00000000  @...............

; ASM-LABEL: amdhsa_kernarg_preload_8_implicit_2:
; ASM: .amdhsa_user_sgpr_count 10
; ASM: .amdhsa_next_free_sgpr 10
; ASM: ; TotalNumSgprs: 16
; ASM: ; NumSGPRsForWavesPerEU: 16

; Only the kernarg_ptr is enabled so we should have 8 preload kernarg SGPRs, 2
; implicit, and 6 extra.

define amdgpu_kernel void @amdhsa_kernarg_preload_8_implicit_2(i256 inreg) #0 { ret void }

; OBJDUMP-NEXT: 0080 00000000 00000000 08010000 00000000  ................
; OBJDUMP-NEXT: 0090 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 00a0 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 00b0 4000af00 86000000 08000100 00000000  @...............

; ASM-LABEL: amdhsa_kernarg_preload_1_implicit_2:
; ASM: .amdhsa_user_sgpr_count 3
; ASM: .amdhsa_next_free_sgpr 3
; ASM: ; TotalNumSgprs: 9
; ASM: ; NumSGPRsForWavesPerEU: 9

; 1 preload, 2 implicit, 6 extra. Rounds up to 16 SGPRs in the KD.

define amdgpu_kernel void @amdhsa_kernarg_preload_1_implicit_2(i32 inreg) #0 { ret void }

; OBJDUMP-NEXT: 00c0 00000000 00000000 08010000 00000000  ................
; OBJDUMP-NEXT: 00d0 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 00e0 00000000 00000000 00000000 00000000  ................
; OBJDUMP-NEXT: 00f0 0000af00 84000000 08000000 00000000  ................

; ASM-LABEL: amdhsa_kernarg_preload_0_implicit_2:
; ASM: .amdhsa_user_sgpr_count 2
; ASM: .amdhsa_next_free_sgpr 0
; ASM: ; TotalNumSgprs: 6
; ASM: ; NumSGPRsForWavesPerEU: 6

; 0 preload kernarg SGPRs, 2 implicit, 6 extra. Rounds up to 8 SGPRs in the KD.
; Encoded like '00'.

define amdgpu_kernel void @amdhsa_kernarg_preload_0_implicit_2(i32) #0 { ret void }

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "uniform-work-group-size"="false" }
