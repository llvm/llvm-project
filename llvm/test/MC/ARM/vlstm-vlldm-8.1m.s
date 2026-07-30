// RUN: llvm-mc -triple=armv8.1m.main-arm-none-eabi -mcpu=generic -show-encoding %s \
// RUN: | FileCheck --check-prefixes=CHECK %s

// RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mcpu=generic -show-encoding %s \
// RUN: | FileCheck --check-prefixes=CHECK %s

vlstm r8, {d0 - d31}
// CHECK: vlstm	r8, {d0 - d31} @ encoding: [0x28,0xec,0x80,0x0a]

vlldm r8, {d0 - d31}
// CHECK: vlldm	r8, {d0 - d31} @ encoding: [0x38,0xec,0x80,0x0a]
