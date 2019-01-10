# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t
# RUN: llvm-objdump -syms -reloc -demangle %t | FileCheck %s

## Check we demangle symbols when printing relocations.
# CHECK:      000000000000001 R_X86_64_PLT32 foo()-4

## Check we demangle symbols when printing symbol table.
# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 0000000000000000 g     F .text           00000000 foo()

## Check the case when relocations are inlined into disassembly.
# RUN: llvm-objdump -d -r -demangle %t | FileCheck %s --check-prefix=INLINE
# INLINE:      Disassembly of section .text:
# INLINE-NEXT: foo():
# INLINE-NEXT:  0: {{.*}}  callq   0 <_Z3foov+0x5>
# INLINE-NEXT:  0000000000000001:  R_X86_64_PLT32 foo()-4

.text
.globl _Z3foov
.type _Z3foov,@function
_Z3foov:
 callq _Z3foov@PLT
