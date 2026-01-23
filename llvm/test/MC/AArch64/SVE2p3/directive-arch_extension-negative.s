// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension sve2p3
.arch_extension nosve2p3
addqp z0.b, z0.b, z0.b
// CHECK: error: instruction requires: sme2p3 or sve2p3
// CHECK-NEXT: addqp z0.b, z0.b, z0.b
