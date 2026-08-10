# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -x .text %t | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym ERR=1 %s -o %t1
# RUN: llvm-readelf -x .text %t1 | FileCheck %s --check-prefix=CHECK1

# CHECK: 0x00000000 9090
# CHECK1: 0x00000000 90909090 90

.subsection 1
661:
  nop
662:
.previous
## 661 and 662 are in the same fragment being laied out.
  .org . - (662b-661b) + (662b-661b)
  nop

.ifdef ERR
.subsection 1
661:
  .p2align 2
  nop
662:
.previous
# ERR: :[[#@LINE+1]]:8: error: expected assembly-time absolute expression
  .org . - (662b-661b) + (662b-661b)
  nop
.endif
