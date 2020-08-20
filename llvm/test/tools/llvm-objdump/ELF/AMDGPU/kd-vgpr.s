; RUN: llvm-mc %s -mattr=+code-object-v3 --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t
; RUN: llvm-objdump --arch-name=amdgcn --mcpu=gfx908 --disassemble-symbols=my_kernel_1.kd %t | tail -n +8 > %t1.s
; RUN: llvm-objdump --arch-name=amdgcn --mcpu=gfx908 --disassemble-symbols=my_kernel_2.kd %t | tail -n +8 > %t2.s
; RUN: llvm-objdump --arch-name=amdgcn --mcpu=gfx908 --disassemble-symbols=my_kernel_3.kd %t | tail -n +8 > %t3.s
; RUN: cat %t1.s %t2.s %t3.s | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t-re-assemble
; RUN: diff %t %t-re-assemble

// Test disassembly for GRANULATED_WORKITEM_VGPR_COUNT.

.amdhsa_kernel my_kernel_1
  .amdhsa_next_free_vgpr 23
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdhsa_kernel my_kernel_2
  .amdhsa_next_free_vgpr 14
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdhsa_kernel my_kernel_3
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

