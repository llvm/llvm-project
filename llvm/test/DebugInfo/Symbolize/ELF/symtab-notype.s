# REQUIRES: x86-registered-target
## STT_NOTYPE symbols are common in assembly files. Test we can symbolize them.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-symbolizer --obj=%t --inlines 0 1 2 3 4 5 6 7 | FileCheck %s
# RUN: llvm-symbolizer --obj=%t --no-inlines 0 1 2 3 4 5 6 7 | FileCheck %s

# CHECK:       _start
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  g_notype
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  g_notype
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:

## This is a gap.
# CHECK-NEXT:  ??
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:

# CHECK-NEXT:  l_notype
# CHECK-NEXT:  symtab-notype.s:0:0
# CHECK-EMPTY:

## TODO addr2line does not symbolize the last two out-of-bounds addresses.
# CHECK-NEXT:  l_notype_nosize
# CHECK-NEXT:  symtab-notype.s:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  l_notype_nosize
# CHECK-NEXT:  symtab-notype.s:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  l_notype_nosize
# CHECK-NEXT:  symtab-notype.s:0:0
# CHECK-EMPTY:

.file "symtab-notype.s"

.globl _start, g_notype
_start:
  retq

g_notype:
  nop
  nop
.size g_notype, . - g_notype

  nop

l_notype:
  nop
.size l_notype, . - l_notype

l_notype_nosize:
  nop
