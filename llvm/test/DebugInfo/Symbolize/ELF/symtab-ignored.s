# REQUIRES: x86-registered-target
## Ignore STT_SECTION and STT_TLS symbols for .symtab symbolization.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-symbolizer --obj=%t 0 | FileCheck %s

# CHECK:       b
# CHECK-NEXT:  1.c:0:0
# CHECK-EMPTY:

.file "1.c"

.section a,"a",@progbits
b:
  .reloc ., R_X86_64_NONE, a
.section c,"a",@progbits
  .reloc ., R_X86_64_NONE, c

.section .tbss,"awT",@nobits
.globl tls
tls:
