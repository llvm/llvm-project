# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-unknown-linux-gnu
# RUN: llvm-as %p/Inputs/multiple-data.ll -o %t2.o
# RUN: %ld_bfd -plugin %llvmshlibdir/LLVMgold%shlibext \
# RUN:     -m elf_x86_64 -o %t.exe %t2.o %t.o  \
# RUN:     --section-ordering-file=%S/Inputs/multiple-data-section-ordering.txt
# RUN: llvm-readelf -s %t.exe | FileCheck %s

# REQUIRES: ld-bfd-supports-section-ordering-file

# CHECK-DAG:      [[#%x, ADDR:]]       4 OBJECT  GLOBAL DEFAULT    2 tin
# CHECK-DAG:      [[#%x, ADDR + 4]]    4 OBJECT  GLOBAL DEFAULT    2 dipsy
# CHECK-DAG:      [[#%x, ADDR + 8]]    4 OBJECT  GLOBAL DEFAULT    2 pat

.globl _start
_start:
  movl $pat, %ecx
  movl $dipsy, %ebx
  movl $tin, %eax
