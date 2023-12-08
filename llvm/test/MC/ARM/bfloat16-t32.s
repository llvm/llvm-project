// RUN: llvm-mc -triple thumbv8 -mattr=+bf16,+neon -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple thumbv8 -mattr=+v8.6a -show-encoding < %s | FileCheck %s  --check-prefix=CHECK

vcvt.bf16.f32 d1, q3
// CHECK:     vcvt.bf16.f32   d1, q3    @ encoding: [0xb6,0xff,0x46,0x16]

it eq
vcvtbeq.bf16.f32  s1, s3
// CHECK: it eq                         @ encoding: [0x08,0xbf]
// CHECK-NEXT: vcvtbeq.bf16.f32 s1, s3  @ encoding:  [0xf3,0xee,0x61,0x09]

it ne
vcvttne.bf16.f32 s1, s3
// CHECK: it ne                         @ encoding: [0x18,0xbf]
// CHECK: vcvttne.bf16.f32 s1, s3       @ encoding: [0xf3,0xee,0xe1,0x09]
