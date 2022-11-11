## Emit ".file" as the source file name when there is no file name.
# RUN: llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=obj -o - | \
# RUN:    llvm-objdump --syms - | FileCheck %s

      .globl .var1
.var1:
      .globl .var2
.var2:

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 00000000      df *DEBUG*	00000000 .file
# CHECK-NEXT: 00000000 l       .text	00000000 .text
# CHECK-NEXT: 00000000 g     F .text (csect: .text) 	00000000 .var1
# CHECK-NEXT: 00000000 g     F .text (csect: .text) 	00000000 .var2
