// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+avxifma -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x95,0xb5,0xe6]
     {vex} vpmadd52huq ymm12, ymm13, ymm14

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x91,0xb5,0xe6]
     {vex} vpmadd52huq xmm12, xmm13, xmm14

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x95,0xb5,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x95,0xb5,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rip]

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0x24,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0xa1,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0xa2,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52huq ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x91,0xb5,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x91,0xb5,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rip]

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0x24,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0xa1,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0xa2,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52huq xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymm14
// CHECK: encoding: [0xc4,0x42,0x95,0xb4,0xe6]
     {vex} vpmadd52luq ymm12, ymm13, ymm14

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmm14
// CHECK: encoding: [0xc4,0x42,0x91,0xb4,0xe6]
     {vex} vpmadd52luq xmm12, xmm13, xmm14

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x95,0xb4,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x95,0xb4,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rip]

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0x24,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0xa1,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0xa2,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52luq ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x91,0xb4,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x91,0xb4,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rip]

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0x24,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0xa1,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0xa2,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52luq xmm12, xmm13, xmmword ptr [rdx - 2048]

