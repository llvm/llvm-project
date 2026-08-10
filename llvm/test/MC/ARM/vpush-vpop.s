@ RUN: llvm-mc -triple armv7-unknown-unknown -mcpu=cortex-a8 -show-encoding < %s | FileCheck --check-prefix=CHECK-ARM %s
@ RUN: llvm-mc -triple thumbv7-unknown-unknown -mcpu=cortex-a8 -show-encoding < %s | FileCheck --check-prefix=CHECK-THUMB %s

foo:
@ CHECK: foo
 vpush {d8, d9, d10, d11, d12}
 vpush {s8, s9, s10, s11, s12}
 vpop  {d8, d9, d10, d11, d12}
 vpop  {s8, s9, s10, s11, s12}
@ optional size suffix
 vpush.s8 {d8, d9, d10, d11, d12}
 vpush.16 {s8, s9, s10, s11, s12}
 vpop.f32  {d8, d9, d10, d11, d12}
 vpop.64  {s8, s9, s10, s11, s12}

@ CHECK-THUMB: vpush {d8, d9, d10, d11, d12} @ encoding: [0x2d,0xed,0x0a,0x8b]
@ CHECK-THUMB: vpush {s8, s9, s10, s11, s12} @ encoding: [0x2d,0xed,0x05,0x4a]
@ CHECK-THUMB: vpop  {d8, d9, d10, d11, d12} @ encoding: [0xbd,0xec,0x0a,0x8b]
@ CHECK-THUMB: vpop  {s8, s9, s10, s11, s12} @ encoding: [0xbd,0xec,0x05,0x4a]

@ CHECK-ARM: vpush {d8, d9, d10, d11, d12} @ encoding: [0x0a,0x8b,0x2d,0xed]
@ CHECK-ARM: vpush {s8, s9, s10, s11, s12} @ encoding: [0x05,0x4a,0x2d,0xed]
@ CHECK-ARM: vpop  {d8, d9, d10, d11, d12} @ encoding: [0x0a,0x8b,0xbd,0xec]
@ CHECK-ARM: vpop  {s8, s9, s10, s11, s12} @ encoding: [0x05,0x4a,0xbd,0xec]

@ CHECK-THUMB: vpush {d8, d9, d10, d11, d12} @ encoding: [0x2d,0xed,0x0a,0x8b]
@ CHECK-THUMB: vpush {s8, s9, s10, s11, s12} @ encoding: [0x2d,0xed,0x05,0x4a]
@ CHECK-THUMB: vpop  {d8, d9, d10, d11, d12} @ encoding: [0xbd,0xec,0x0a,0x8b]
@ CHECK-THUMB: vpop  {s8, s9, s10, s11, s12} @ encoding: [0xbd,0xec,0x05,0x4a]

@ CHECK-ARM: vpush {d8, d9, d10, d11, d12} @ encoding: [0x0a,0x8b,0x2d,0xed]
@ CHECK-ARM: vpush {s8, s9, s10, s11, s12} @ encoding: [0x05,0x4a,0x2d,0xed]
@ CHECK-ARM: vpop  {d8, d9, d10, d11, d12} @ encoding: [0x0a,0x8b,0xbd,0xec]
@ CHECK-ARM: vpop  {s8, s9, s10, s11, s12} @ encoding: [0x05,0x4a,0xbd,0xec]
