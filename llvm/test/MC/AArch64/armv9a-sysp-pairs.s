// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+d128,+tlb-rmi < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ASM,CHECK-ENCODING
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+d128,+tlb-rmi < %s \
// RUN:   | llvm-objdump -d --mattr=-d128,-tlb-rmi --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIS

sysp #0, c8, c2, #2, x8, x9
// CHECK-ASM: sysp #0, c8, c2, #2, x8, x9
// CHECK-ENCODING: encoding: [0x48,0x82,0x48,0xd5]
// CHECK-DIS: sysp #0, c8, c2, #2, x8, x9

sysp #0, c8, c2, #2, xzr, xzr
// CHECK-ASM: sysp #0, c8, c2, #2
// CHECK-ENCODING: encoding: [0x5f,0x82,0x48,0xd5]
// CHECK-DIS: sysp #0, c8, c2, #2

tlbip VAE1, x8, x9
// CHECK-ASM: tlbip vae1, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x87,0x48,0xd5]
// CHECK-DIS: sysp #0, c8, c7, #1, x8, x9

tlbip VAE1, xzr, xzr
// CHECK-ASM: tlbip vae1, xzr, xzr
// CHECK-ENCODING: encoding: [0x3f,0x87,0x48,0xd5]
// CHECK-DIS: sysp #0, c8, c7, #1
