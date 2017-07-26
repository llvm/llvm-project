# RUN: echo ".data.tin" > %t_order_lto.txt
# RUN: echo ".data.dipsy" >> %t_order_lto.txt
# RUN: echo ".data.pat" >> %t_order_lto.txt

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-unknown-linux-gnu
# RUN: llvm-as %p/Inputs/multiple-data.ll -o %t2.o
# RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
# RUN:     -m elf_x86_64 -o %t.exe %t2.o %t.o  \
# RUN:     --section-ordering-file=%t_order_lto.txt
# RUN: llvm-readobj -elf-output-style=GNU -t %t.exe | FileCheck %s

# CHECK: Symbol table '.symtab' contains 9 entries:
# CHECK-NEXT:    Num:    Value          Size Type    Bind   Vis      Ndx Name
# CHECK-NEXT:      0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
# CHECK-NEXT:      1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS ld-temp.o
# CHECK-NEXT:      2: 0000000000401104     0 NOTYPE  GLOBAL DEFAULT  ABS _end
# CHECK-NEXT:      3: 0000000000401104     0 NOTYPE  GLOBAL DEFAULT  ABS __bss_start
# CHECK-NEXT:      4: 0000000000401104     0 NOTYPE  GLOBAL DEFAULT  ABS _edata
# CHECK-NEXT:      5: 00000000004000e8     0 NOTYPE  GLOBAL DEFAULT    1 _start
# CHECK-NEXT:      6: 00000000004010fc     4 OBJECT  GLOBAL DEFAULT    2 dipsy
# CHECK-NEXT:      7: 00000000004010f8     4 OBJECT  GLOBAL DEFAULT    2 tin
# CHECK-NEXT:      8: 0000000000401100     4 OBJECT  GLOBAL DEFAULT    2 pat

.globl _start
_start:
  movl $pat, %ecx
  movl $dipsy, %ebx
  movl $tin, %eax
