// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2>&1 | FileCheck %s
// CHECK: LLVM ERROR: Cannot encode high byte register in REX-prefixed instruction
 movzx %dh, %rsi
