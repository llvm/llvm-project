# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - | FileCheck %s
# CHECK: .debug_addr contents:
# CHECK-NOT: Addr
# CHECK-NOT: error:

.section .debug_addr,"",@progbits
