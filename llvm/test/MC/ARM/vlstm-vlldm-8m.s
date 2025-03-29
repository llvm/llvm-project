// RUN: llvm-mc -triple=armv8m.main-arm-none-eabi -mcpu=generic -show-encoding %s \
// RUN: | FileCheck --check-prefixes=CHECK %s

// RUN: llvm-mc -triple=thumbv8m.main-none-eabi -mcpu=generic -show-encoding %s \
// RUN: | FileCheck --check-prefixes=CHECK %s

vlstm r8, {d0 - d15}
// CHECK: vlstm	r8, {d0 - d15} @ encoding: [0x28,0xec,0x00,0x0a]

vlldm r8, {d0 - d15}
// CHECK: vlldm	r8, {d0 - d15} @ encoding: [0x38,0xec,0x00,0x0a]

vlstm r8
// CHECK: vlstm	r8, {d0 - d15} @ encoding: [0x28,0xec,0x00,0x0a]

vlldm r8
// CHECK: vlldm r8, {d0 - d15} @ encoding: [0x38,0xec,0x00,0x0a]
