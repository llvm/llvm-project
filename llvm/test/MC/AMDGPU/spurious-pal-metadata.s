# RUN: llvm-mc -triple=amdgcn--amdhsa -mcpu=fiji %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=amdgcn--amdhsa -mcpu=fiji %s -o %t.o
# RUN: llvm-objdump -s %t.o | FileCheck %s --check-prefix=OBJDUMP

# Check that we don't get spurious PAL metadata. 

# ASM-NOT: pal_metadata
# OBJDUMP-NOT: section .note
