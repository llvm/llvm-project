# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t

# RUN: llvm-objdump --section-headers %t | FileCheck %s
# CHECK:     Idx Name
# CHECK:      3  .foo
# CHECK-NEXT: 4  .bar
# CHECK-NEXT: 5  .zed

## Check we report the valid section index
## when requesting a specific section.
# RUN: llvm-objdump --section-headers --section=.bar %t \
# RUN:   | FileCheck %s --check-prefix=BAR
# BAR:      Idx Name
# BAR-NEXT:  4  .bar
# BAR-NOT:  foo
# BAR-NOT:  zed

.section .foo, "ax", %progbits
nop

.section .bar, "ax", %progbits
nop

.section .zed, "ax", %progbits
nop
