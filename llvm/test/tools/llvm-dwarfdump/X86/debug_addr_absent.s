# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2>&1 | FileCheck %s
# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
