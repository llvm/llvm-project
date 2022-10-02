# RUN: llvm-mc %s -filetype=obj -triple=i386 -o %t.32.o
# RUN: llvm-mc %s -filetype=obj -triple=x86_64 -o %t.64.o

## Check we print the address of `foo` and `bar`.
# RUN: llvm-objdump -D %t.32.o | FileCheck --check-prefixes=ADDR32,ADDR %s --match-full-lines --strict-whitespace
# RUN: llvm-objdump -D %t.64.o | FileCheck --check-prefixes=ADDR64,ADDR %s --match-full-lines --strict-whitespace
#    ADDR32:00000000 <foo>:
#    ADDR64:0000000000000000 <foo>:
# ADDR-NEXT:       0: 90{{ +}}	nop
# ADDR-NEXT:       1: 90{{ +}}	nop
#ADDR-EMPTY:
#    ADDR32:00000002 <bar>:
#    ADDR64:0000000000000002 <bar>:
# ADDR-NEXT:       2: 90{{ +}}	nop
#      ADDR:Disassembly of section .data:
#ADDR-EMPTY:
#    ADDR32:00000000 <.data>:
#    ADDR64:0000000000000000 <.data>:
# ADDR-NEXT:       0: 01 00{{ +}}	addl	%eax, (%{{[er]}}ax)
#ADDR-EMPTY:

## Check we do not print the addresses with --no-leading-addr.
# RUN: llvm-objdump -d --no-leading-addr %t.32.o > %t.32.txt
# RUN: llvm-objdump -d --no-leading-addr %t.64.o > %t.64.txt
# RUN: FileCheck --input-file=%t.32.txt %s --check-prefix=NOADDR --match-full-lines --strict-whitespace
# RUN: FileCheck --input-file=%t.64.txt %s --check-prefix=NOADDR --match-full-lines --strict-whitespace

#      NOADDR:<foo>:
# NOADDR-NEXT: 90{{ +}}	nop
# NOADDR-NEXT: 90{{ +}}	nop
#NOADDR-EMPTY:
#      NOADDR:<bar>:
# NOADDR-NEXT: 90{{ +}}	nop

.text
.globl  foo
.type   foo, @function
foo:
 nop
 nop

bar:
 nop

.data
  .word 1
