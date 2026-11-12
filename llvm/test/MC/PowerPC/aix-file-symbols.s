## Check mutiple C_FILE symbols are emitted.
# RUN: llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=obj -o - | \
# RUN:    llvm-readobj --syms - | FileCheck %s

      .file   "1.c"
      .globl .var1
.var1:
      .file   "2.c"
      .globl .var2
.var2:
      .file   "3.c"

# CHECK:      Symbols [
# CHECK:        Name: 1.c
# CHECK-NEXT:   Type: XFT_FN (0x0)
# CHECK:        Name: 2.c
# CHECK-NEXT:   Type: XFT_FN (0x0)
# CHECK:        Name: 3.c
# CHECK-NEXT:   Type: XFT_FN (0x0)
