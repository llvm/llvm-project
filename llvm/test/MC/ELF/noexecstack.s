# RUN: llvm-mc -filetype=obj -triple x86_64 %s -no-exec-stack | llvm-readelf -S - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64 %s | llvm-readelf -S - | FileCheck %s --check-prefix=NOSTACK

## Solaris doesn't use .note.GNU-stack at all.
# RUN: llvm-mc -filetype=obj -triple x86_64-solaris %s | llvm-readelf -S - | FileCheck %s --check-prefix=NOSTACK

# CHECK: .text             PROGBITS        0000000000000000 {{[0-9a-f]+}} 000001 00  AX  0   0  4
# CHECK: .note.GNU-stack   PROGBITS        0000000000000000 {{[0-9a-f]+}} 000000 00      0   0  1

# NOSTACK-NOT: .note.GNU-stack
nop
