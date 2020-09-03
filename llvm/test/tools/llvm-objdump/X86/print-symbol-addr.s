# RUN: llvm-mc %s -filetype=obj -triple=i386 -o %t.32.o
# RUN: llvm-mc %s -filetype=obj -triple=x86_64 -o %t.64.o

## Check we print the address of `foo` and `bar`.
# RUN: llvm-objdump -d %t.32.o | FileCheck --check-prefixes=ADDR32,ADDR %s
# RUN: llvm-objdump -d %t.64.o | FileCheck --check-prefixes=ADDR64,ADDR %s
# ADDR32:    00000000 <foo>:
# ADDR64:    0000000000000000 <foo>:
# ADDR-NEXT:   0: {{.*}}  nop
# ADDR-NEXT:   1: {{.*}}  nop
# ADDR32:    00000002 <bar>:
# ADDR64:    0000000000000002 <bar>:
# ADDR-NEXT:   2: {{.*}}  nop

## Check we do not print the addresses with --no-leading-addr.
# RUN: llvm-objdump -d --no-leading-addr %t.32.o | FileCheck %s --check-prefix=NOADDR
# RUN: llvm-objdump -d --no-leading-addr %t.64.o | FileCheck %s --check-prefix=NOADDR
# NOADDR:      <foo>:
# NOADDR-NEXT:   {{.*}} nop
# NOADDR-NEXT:   {{.*}} nop
# NOADDR:      <bar>:
# NOADDR-NEXT:   {{.*}} nop

.text
.globl  foo
.type   foo, @function
foo:
 nop
 nop

bar:
 nop
