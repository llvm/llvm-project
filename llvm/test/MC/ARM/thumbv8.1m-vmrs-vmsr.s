// RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+8msecext,+mve -show-encoding < %s \
// RUN: | FileCheck --check-prefix=CHECK %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=-8msecext,+mve -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK-NOSEC %s
// RUN:   FileCheck --check-prefix=ERROR-NOSEC < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+8msecext,-mve,+vfp2 -show-encoding < %s 2> %t \
// RUN: | FileCheck --check-prefix=CHECK-NOMVE %s
// RUN: FileCheck --check-prefix=ERROR-NOMVE < %t %s
// RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+8msecext,+mve,-vfp2 -show-encoding < %s \
// RUN: | FileCheck --check-prefix=CHECK-NOVFP %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=-8msecext,-mve,-vfp2 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=ERROR-NONE < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main-none-eabi -mattr=+8msecext,+vfp2 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=ERROR-V8M < %t %s

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOVFP: vmsr fpscr_nzcvqc, r0              @ encoding: [0xe2,0xee,0x10,0x0a]
// CHECK-NOMVE: vmsr fpscr_nzcvqc, r0              @ encoding: [0xe2,0xee,0x10,0x0a]
// CHECK-NOSEC: vmsr fpscr_nzcvqc, r0              @ encoding: [0xe2,0xee,0x10,0x0a]
// CHECK: vmsr fpscr_nzcvqc, r0              @ encoding: [0xe2,0xee,0x10,0x0a]
vmsr fpscr_nzcvqc, r0

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: fp registers
// CHECK-NOVFP: vmrs r10, fpscr_nzcvqc              @ encoding: [0xf2,0xee,0x10,0xaa]
// CHECK-NOMVE: vmrs r10, fpscr_nzcvqc              @ encoding: [0xf2,0xee,0x10,0xaa]
// CHECK-NOSEC: vmrs r10, fpscr_nzcvqc              @ encoding: [0xf2,0xee,0x10,0xaa]
// CHECK: vmrs r10, fpscr_nzcvqc              @ encoding: [0xf2,0xee,0x10,0xaa]
vmrs r10, fpscr_nzcvqc

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// CHECK-NOVFP: vmrs r0, fpcxtns              @ encoding: [0xfe,0xee,0x10,0x0a]
// CHECK-NOMVE: vmrs r0, fpcxtns              @ encoding: [0xfe,0xee,0x10,0x0a]
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK: vmrs r0, fpcxtns              @ encoding: [0xfe,0xee,0x10,0x0a]
vmrs r0, fpcxtns

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// CHECK-NOVFP: vmsr fpcxtns, r10              @ encoding: [0xee,0xee,0x10,0xaa]
// CHECK-NOMVE: vmsr fpcxtns, r10              @ encoding: [0xee,0xee,0x10,0xaa]
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK: vmsr fpcxtns, r10              @ encoding: [0xee,0xee,0x10,0xaa]
vmsr fpcxtns, r10

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// CHECK-NOVFP: vmsr fpcxts, r5              @ encoding: [0xef,0xee,0x10,0x5a]
// CHECK-NOMVE: vmsr fpcxts, r5              @ encoding: [0xef,0xee,0x10,0x5a]
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK: vmsr fpcxts, r5              @ encoding: [0xef,0xee,0x10,0x5a]
vmsr fpcxts, r5

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// CHECK-NOVFP: vmrs  r3, fpcxtns              @ encoding: [0xfe,0xee,0x10,0x3a]
// CHECK-NOMVE: vmrs  r3, fpcxtns              @ encoding: [0xfe,0xee,0x10,0x3a]
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK: vmrs  r3, fpcxtns              @ encoding: [0xfe,0xee,0x10,0x3a]
vmrs r3, fpcxtns

// ERROR-V8M: instruction requires: armv8.1m.main
// ERROR-NONE: instruction requires: ARMv8-M Security Extensions
// CHECK-NOVFP: vmrs  r0, fpcxts              @ encoding: [0xff,0xee,0x10,0x0a]
// CHECK-NOMVE: vmrs  r0, fpcxts              @ encoding: [0xff,0xee,0x10,0x0a]
// ERROR-NOSEC: instruction requires: ARMv8-M Security Extensions
// CHECK: vmrs  r0, fpcxts              @ encoding: [0xff,0xee,0x10,0x0a]
vmrs r0, fpcxts

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOSEC: vmrs r0, vpr              @ encoding: [0xfc,0xee,0x10,0x0a]
// CHECK: vmrs r0, vpr              @ encoding: [0xfc,0xee,0x10,0x0a]
vmrs r0, vpr

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOSEC: vmrs r4, p0              @ encoding: [0xfd,0xee,0x10,0x4a]
// CHECK: vmrs r4, p0              @ encoding: [0xfd,0xee,0x10,0x4a]
vmrs r4, p0

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOSEC: vmsr vpr, r0              @ encoding: [0xec,0xee,0x10,0x0a]
// CHECK: vmsr vpr, r0              @ encoding: [0xec,0xee,0x10,0x0a]
vmsr vpr, r0

// ERROR-V8M: instruction requires: mve armv8.1m.main
// ERROR-NONE: instruction requires: mve
// ERROR-NOMVE: instruction requires: mve
// CHECK-NOSEC: vmsr p0, r4              @ encoding: [0xed,0xee,0x10,0x4a]
// CHECK: vmsr p0, r4              @ encoding: [0xed,0xee,0x10,0x4a]
vmsr p0, r4
