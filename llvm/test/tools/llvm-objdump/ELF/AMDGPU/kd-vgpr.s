;; Test disassembly for GRANULATED_WORKITEM_VGPR_COUNT in the kernel descriptor.

; RUN: split-file %s %t.dir

; RUN: llvm-mc %t.dir/1.s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t1
; RUN: llvm-objdump --disassemble-symbols=my_kernel_1.kd %t1 | tail -n +8 \
; RUN: | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t1-re-assemble
; RUN: diff %t1 %t1-re-assemble

; RUN: llvm-mc %t.dir/2.s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t2
; RUN: llvm-objdump --disassemble-symbols=my_kernel_2.kd %t2 | tail -n +8 \
; RUN: | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t2-re-assemble
; RUN: diff %t2 %t2-re-assemble

; RUN: llvm-mc %t.dir/3.s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t3
; RUN: llvm-objdump --disassemble-symbols=my_kernel_3.kd %t3 | tail -n +8 \
; RUN: | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t3-re-assemble
; RUN: diff %t3 %t3-re-assemble

;--- 1.s
.amdhsa_kernel my_kernel_1
  .amdhsa_next_free_vgpr 23
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

;--- 2.s
.amdhsa_kernel my_kernel_2
  .amdhsa_next_free_vgpr 14
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

;--- 3.s
.amdhsa_kernel my_kernel_3
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel
