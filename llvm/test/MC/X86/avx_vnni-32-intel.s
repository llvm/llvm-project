// RUN: llvm-mc -triple i686-unknown-unknown -mattr=+avxvnni -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymm4
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xf4]
          {vex} vpdpbusd ymm6, ymm5, ymm4

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmm4
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xf4]
          {vex} vpdpbusd xmm6, xmm5, xmm4

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusd ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusd ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0x30]
          {vex} vpdpbusd ymm6, ymm5, ymmword ptr [eax]

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpbusd ymm6, ymm5, ymmword ptr [2*ebp - 1024]

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpbusd ymm6, ymm5, ymmword ptr [ecx + 4064]

// CHECK: {vex} vpdpbusd ymm6, ymm5, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x55,0x50,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpbusd ymm6, ymm5, ymmword ptr [edx - 4096]

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusd xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusd xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0x30]
          {vex} vpdpbusd xmm6, xmm5, xmmword ptr [eax]

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpbusd xmm6, xmm5, xmmword ptr [2*ebp - 512]

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpbusd xmm6, xmm5, xmmword ptr [ecx + 2032]

// CHECK: {vex} vpdpbusd xmm6, xmm5, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x51,0x50,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpbusd xmm6, xmm5, xmmword ptr [edx - 2048]

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymm4
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xf4]
          {vex} vpdpbusds ymm6, ymm5, ymm4

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmm4
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xf4]
          {vex} vpdpbusds xmm6, xmm5, xmm4

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusds ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusds ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0x30]
          {vex} vpdpbusds ymm6, ymm5, ymmword ptr [eax]

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpbusds ymm6, ymm5, ymmword ptr [2*ebp - 1024]

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpbusds ymm6, ymm5, ymmword ptr [ecx + 4064]

// CHECK: {vex} vpdpbusds ymm6, ymm5, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x55,0x51,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpbusds ymm6, ymm5, ymmword ptr [edx - 4096]

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpbusds xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpbusds xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0x30]
          {vex} vpdpbusds xmm6, xmm5, xmmword ptr [eax]

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpbusds xmm6, xmm5, xmmword ptr [2*ebp - 512]

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpbusds xmm6, xmm5, xmmword ptr [ecx + 2032]

// CHECK: {vex} vpdpbusds xmm6, xmm5, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x51,0x51,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpbusds xmm6, xmm5, xmmword ptr [edx - 2048]

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymm4
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xf4]
          {vex} vpdpwssd ymm6, ymm5, ymm4

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmm4
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xf4]
          {vex} vpdpwssd xmm6, xmm5, xmm4

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssd ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssd ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0x30]
          {vex} vpdpwssd ymm6, ymm5, ymmword ptr [eax]

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpwssd ymm6, ymm5, ymmword ptr [2*ebp - 1024]

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpwssd ymm6, ymm5, ymmword ptr [ecx + 4064]

// CHECK: {vex} vpdpwssd ymm6, ymm5, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x55,0x52,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpwssd ymm6, ymm5, ymmword ptr [edx - 4096]

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssd xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssd xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0x30]
          {vex} vpdpwssd xmm6, xmm5, xmmword ptr [eax]

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpwssd xmm6, xmm5, xmmword ptr [2*ebp - 512]

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpwssd xmm6, xmm5, xmmword ptr [ecx + 2032]

// CHECK: {vex} vpdpwssd xmm6, xmm5, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x51,0x52,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpwssd xmm6, xmm5, xmmword ptr [edx - 2048]

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymm4
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xf4]
          {vex} vpdpwssds ymm6, ymm5, ymm4

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmm4
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xf4]
          {vex} vpdpwssds xmm6, xmm5, xmm4

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssds ymm6, ymm5, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssds ymm6, ymm5, ymmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0x30]
          {vex} vpdpwssds ymm6, ymm5, ymmword ptr [eax]

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0x34,0x6d,0x00,0xfc,0xff,0xff]
          {vex} vpdpwssds ymm6, ymm5, ymmword ptr [2*ebp - 1024]

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb1,0xe0,0x0f,0x00,0x00]
          {vex} vpdpwssds ymm6, ymm5, ymmword ptr [ecx + 4064]

// CHECK: {vex} vpdpwssds ymm6, ymm5, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0x55,0x53,0xb2,0x00,0xf0,0xff,0xff]
          {vex} vpdpwssds ymm6, ymm5, ymmword ptr [edx - 4096]

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb4,0xf4,0x00,0x00,0x00,0x10]
          {vex} vpdpwssds xmm6, xmm5, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb4,0x87,0x23,0x01,0x00,0x00]
          {vex} vpdpwssds xmm6, xmm5, xmmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0x30]
          {vex} vpdpwssds xmm6, xmm5, xmmword ptr [eax]

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0x34,0x6d,0x00,0xfe,0xff,0xff]
          {vex} vpdpwssds xmm6, xmm5, xmmword ptr [2*ebp - 512]

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb1,0xf0,0x07,0x00,0x00]
          {vex} vpdpwssds xmm6, xmm5, xmmword ptr [ecx + 2032]

// CHECK: {vex} vpdpwssds xmm6, xmm5, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0x51,0x53,0xb2,0x00,0xf8,0xff,0xff]
          {vex} vpdpwssds xmm6, xmm5, xmmword ptr [edx - 2048]

