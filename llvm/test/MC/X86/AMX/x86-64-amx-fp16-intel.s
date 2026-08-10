// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      tdpfp16ps       tmm3, tmm4, tmm5
// CHECK: encoding: [0xc4,0xe2,0x53,0x5c,0xdc]
               tdpfp16ps       tmm3, tmm4, tmm5
