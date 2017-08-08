// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512vl -mattr=+avx512dq -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:  vcvtps2qq xmm2 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x7b,0x51,0x10]
          vcvtps2qq xmm2 {k2} {z}, qword ptr [rcx + 0x80]

// CHECK:  vcvtps2qq xmm2 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x7b,0x51,0x10]
          vcvtps2qq xmm2 {k2}, qword ptr [rcx + 0x80]

// CHECK:  vcvtps2qq xmm2, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x7b,0x51,0x10]
          vcvtps2qq xmm2, qword ptr [rcx + 0x80]

// CHECK:  vcvttps2qq xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x7a,0x49,0x10]
          vcvttps2qq xmm1 {k2} {z}, qword ptr [rcx + 0x80]

// CHECK:  vcvttps2qq xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x7a,0x49,0x10]
          vcvttps2qq xmm1 {k2}, qword ptr [rcx + 0x80]

// CHECK:  vcvttps2qq xmm1, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x7a,0x49,0x10]
          vcvttps2qq xmm1, qword ptr [rcx + 0x80]

// CHECK:  vcvtps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x79,0x49,0x10]
          vcvtps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]

// CHECK:  vcvtps2uqq xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x79,0x49,0x10]
          vcvtps2uqq xmm1 {k2}, qword ptr [rcx + 128]

// CHECK:  vcvtps2uqq xmm1, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x79,0x49,0x10]
          vcvtps2uqq xmm1, qword ptr [rcx + 128]

// CHECK:  vcvttps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x78,0x49,0x10]
          vcvttps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]

// CHECK:  vcvttps2uqq xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
          vcvttps2uqq xmm1 {k2}, qword ptr [rcx + 128]

// CHECK:  vcvttps2uqq xmm1, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x78,0x49,0x10]
          vcvttps2uqq xmm1, qword ptr [rcx + 128]
// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512vl -mattr=+avx512dq -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:  vcvtps2qq xmm2 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x7b,0x51,0x10]
          vcvtps2qq xmm2 {k2} {z}, qword ptr [rcx + 0x80]

// CHECK:  vcvtps2qq xmm2 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x7b,0x51,0x10]
          vcvtps2qq xmm2 {k2}, qword ptr [rcx + 0x80]

// CHECK:  vcvtps2qq xmm2, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x7b,0x51,0x10]
          vcvtps2qq xmm2, qword ptr [rcx + 0x80]

// CHECK:  vcvttps2qq xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x7a,0x49,0x10]
          vcvttps2qq xmm1 {k2} {z}, qword ptr [rcx + 0x80]

// CHECK:  vcvttps2qq xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x7a,0x49,0x10]
          vcvttps2qq xmm1 {k2}, qword ptr [rcx + 0x80]

// CHECK:  vcvttps2qq xmm1, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x7a,0x49,0x10]
          vcvttps2qq xmm1, qword ptr [rcx + 0x80]

// CHECK:  vcvtps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x79,0x49,0x10]
          vcvtps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]

// CHECK:  vcvtps2uqq xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x79,0x49,0x10]
          vcvtps2uqq xmm1 {k2}, qword ptr [rcx + 128]

// CHECK:  vcvtps2uqq xmm1, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x79,0x49,0x10]
          vcvtps2uqq xmm1, qword ptr [rcx + 128]

// CHECK:  vcvttps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x8a,0x78,0x49,0x10]
          vcvttps2uqq xmm1 {k2} {z}, qword ptr [rcx + 128]

// CHECK:  vcvttps2uqq xmm1 {k2}, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
          vcvttps2uqq xmm1 {k2}, qword ptr [rcx + 128]

// CHECK:  vcvttps2uqq xmm1, qword ptr [rcx + 128]
// CHECK:  encoding: [0x62,0xf1,0x7d,0x08,0x78,0x49,0x10]
          vcvttps2uqq xmm1, qword ptr [rcx + 128]
