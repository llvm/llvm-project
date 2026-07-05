# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN: llvm-dwarfdump --debug-rnglists - | FileCheck %s
# CHECK: .debug_rnglists contents:
# CHECK-NOT: Range List Header:
# CHECK-NOT: error:

.section .debug_rnglists,"",@progbits
