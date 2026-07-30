# RUN: echo ".data.tin" > %t_order_lto.txt
# RUN: echo ".data.dipsy" >> %t_order_lto.txt
# RUN: echo ".data.pat" >> %t_order_lto.txt

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-unknown-linux-gnu
# RUN: llvm-as %p/Inputs/multiple-data.ll -o %t2.o
# RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
# RUN:     -m elf_x86_64 -o %t.exe %t2.o %t.o  \
# RUN:     --section-ordering-file=%t_order_lto.txt
# RUN: llvm-readelf -s %t.exe | FileCheck %s

# CHECK-DAG:      00000000004010fc     4 OBJECT  GLOBAL DEFAULT    2 dipsy
# CHECK-DAG:      00000000004010f8     4 OBJECT  GLOBAL DEFAULT    2 tin
# CHECK-DAG:      0000000000401100     4 OBJECT  GLOBAL DEFAULT    2 pat

.globl _start
_start:
  movl $pat, %ecx
  movl $dipsy, %ebx
  movl $tin, %eax
