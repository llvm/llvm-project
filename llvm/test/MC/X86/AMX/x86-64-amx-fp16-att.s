// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK:      tdpfp16ps       %tmm5, %tmm4, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x53,0x5c,0xdc]
               tdpfp16ps       %tmm5, %tmm4, %tmm3
