## Check mutiple C_FILE symbols are emitted.
# RUN: llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=obj -o - | \
# RUN:    llvm-objdump --syms - | FileCheck %s

      .file   "1.c"
      .globl .var1
.var1:
      .file   "2.c"
      .globl .var2
.var2:
      .file   "3.c"

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 00000000      df *DEBUG*	00000000 1.c
# CHECK-NEXT: 00000000      df *DEBUG*	00000000 2.c
# CHECK-NEXT: 00000000      df *DEBUG*	00000000 3.c
# CHECK-NEXT: 00000000 l       .text	00000000 .text
# CHECK-NEXT: 00000000 g     F .text (csect: .text) 	00000000 .var1
# CHECK-NEXT: 00000000 g     F .text (csect: .text) 	00000000 .var2
