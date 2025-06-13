// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      vsm4key4 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0xd4]
               vsm4key4 zmm2, zmm3, zmm4

// CHECK:      vsm4key4 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               vsm4key4 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK:      vsm4key4 zmm2, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               vsm4key4 zmm2, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK:      vsm4key4 zmm2, zmm3, zmmword ptr [eax]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x10]
               vsm4key4 zmm2, zmm3, zmmword ptr [eax]

// CHECK:      vsm4key4 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x14,0x6d,0x00,0xf8,0xff,0xff]
               vsm4key4 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK:      vsm4key4 zmm2, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x51,0x7f]
               vsm4key4 zmm2, zmm3, zmmword ptr [ecx + 8128]

// CHECK:      vsm4key4 zmm2, zmm3, zmmword ptr [edx - 8192]
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x52,0x80]
               vsm4key4 zmm2, zmm3, zmmword ptr [edx - 8192]

// CHECK:      vsm4rnds4 zmm2, zmm3, zmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0xd4]
               vsm4rnds4 zmm2, zmm3, zmm4

// CHECK:      vsm4rnds4 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               vsm4rnds4 zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456]

// CHECK:      vsm4rnds4 zmm2, zmm3, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               vsm4rnds4 zmm2, zmm3, zmmword ptr [edi + 4*eax + 291]

// CHECK:      vsm4rnds4 zmm2, zmm3, zmmword ptr [eax]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x10]
               vsm4rnds4 zmm2, zmm3, zmmword ptr [eax]

// CHECK:      vsm4rnds4 zmm2, zmm3, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x14,0x6d,0x00,0xf8,0xff,0xff]
               vsm4rnds4 zmm2, zmm3, zmmword ptr [2*ebp - 2048]

// CHECK:      vsm4rnds4 zmm2, zmm3, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x51,0x7f]
               vsm4rnds4 zmm2, zmm3, zmmword ptr [ecx + 8128]

// CHECK:      vsm4rnds4 zmm2, zmm3, zmmword ptr [edx - 8192]
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x52,0x80]
               vsm4rnds4 zmm2, zmm3, zmmword ptr [edx - 8192]

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0xd4]
               {evex} vsm4key4 ymm2, ymm3, ymm4

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0xd4]
               {evex} vsm4key4 xmm2, xmm3, xmm4

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4key4 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4key4 ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x10]
               {evex} vsm4key4 ymm2, ymm3, ymmword ptr [eax]

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x14,0x6d,0x00,0xfc,0xff,0xff]
               {evex} vsm4key4 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x51,0x7f]
               {evex} vsm4key4 ymm2, ymm3, ymmword ptr [ecx + 4064]

// CHECK:      {evex} vsm4key4 ymm2, ymm3, ymmword ptr [edx - 4096]
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x52,0x80]
               {evex} vsm4key4 ymm2, ymm3, ymmword ptr [edx - 4096]

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4key4 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4key4 xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x10]
               {evex} vsm4key4 xmm2, xmm3, xmmword ptr [eax]

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x14,0x6d,0x00,0xfe,0xff,0xff]
               {evex} vsm4key4 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x51,0x7f]
               {evex} vsm4key4 xmm2, xmm3, xmmword ptr [ecx + 2032]

// CHECK:      {evex} vsm4key4 xmm2, xmm3, xmmword ptr [edx - 2048]
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x52,0x80]
               {evex} vsm4key4 xmm2, xmm3, xmmword ptr [edx - 2048]

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymm4
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0xd4]
               {evex} vsm4rnds4 ymm2, ymm3, ymm4

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmm4
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0xd4]
               {evex} vsm4rnds4 xmm2, xmm3, xmm4

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x10]
               {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [eax]

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x14,0x6d,0x00,0xfc,0xff,0xff]
               {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x51,0x7f]
               {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [ecx + 4064]

// CHECK:      {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [edx - 4096]
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x52,0x80]
               {evex} vsm4rnds4 ymm2, ymm3, ymmword ptr [edx - 4096]

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x10]
               {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [eax]

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x14,0x6d,0x00,0xfe,0xff,0xff]
               {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x51,0x7f]
               {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [ecx + 2032]

// CHECK:      {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [edx - 2048]
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x52,0x80]
               {evex} vsm4rnds4 xmm2, xmm3, xmmword ptr [edx - 2048]
