# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -x .text %t | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple=x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# CHECK: 0x00000000 9090

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
