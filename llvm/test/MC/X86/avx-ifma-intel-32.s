// RUN: llvm-mc -triple i686-unknown-unknown -mattr=+avxifma -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0xd4]
     {vex} vpmadd52huq ymm2, ymm3, ymm4

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0xd4]
     {vex} vpmadd52huq xmm2, xmm3, xmm4

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x10]
     {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [eax]

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x14,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x91,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [ecx + 4064]

// CHECK: {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb5,0x92,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52huq ymm2, ymm3, ymmword ptr [edx - 4096]

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x10]
     {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [eax]

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x14,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x91,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [ecx + 2032]

// CHECK: {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb5,0x92,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52huq xmm2, xmm3, xmmword ptr [edx - 2048]

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymm4
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0xd4]
     {vex} vpmadd52luq ymm2, ymm3, ymm4

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmm4
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0xd4]
     {vex} vpmadd52luq xmm2, xmm3, xmm4

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x10]
     {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [eax]

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x14,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [2*ebp - 1024]

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x91,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [ecx + 4064]

// CHECK: {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [edx - 4096]
// CHECK: encoding: [0xc4,0xe2,0xe5,0xb4,0x92,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52luq ymm2, ymm3, ymmword ptr [edx - 4096]

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x94,0xf4,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x94,0x87,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [edi + 4*eax + 291]

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [eax]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x10]
     {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [eax]

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x14,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [2*ebp - 512]

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x91,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [ecx + 2032]

// CHECK: {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [edx - 2048]
// CHECK: encoding: [0xc4,0xe2,0xe1,0xb4,0x92,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52luq xmm2, xmm3, xmmword ptr [edx - 2048]

