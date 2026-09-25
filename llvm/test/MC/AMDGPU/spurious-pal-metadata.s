# RUN: llvm-mc -triple=amdgpu8.03--amdhsa %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=amdgpu8.03--amdhsa %s -o %t.o
# RUN: llvm-objdump -s %t.o | FileCheck %s --check-prefix=OBJDUMP

# Check that we don't get spurious PAL metadata. 

# ASM-NOT: pal_metadata
# OBJDUMP-NOT: section .note
