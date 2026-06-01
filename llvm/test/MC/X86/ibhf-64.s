// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s 

// CHECK: ibhf
// CHECK: encoding: [0xf3,0x48,0x0f,0x1e,0xf8]
ibhf