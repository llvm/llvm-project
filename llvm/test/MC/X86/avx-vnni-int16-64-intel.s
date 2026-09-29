// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vpdpwsud ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x16,0xd2,0xe4]
          vpdpwsud ymm12, ymm13, ymm4

// CHECK: vpdpwsud xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x12,0xd2,0xe4]
          vpdpwsud xmm12, xmm13, xmm4

// CHECK: vpdpwsud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x16,0xd2,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwsud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x16,0xd2,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwsud ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x16,0xd2,0x25,0x00,0x00,0x00,0x00]
          vpdpwsud ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpwsud ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x16,0xd2,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwsud ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpwsud ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x16,0xd2,0xa1,0xe0,0x0f,0x00,0x00]
          vpdpwsud ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vpdpwsud ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x16,0xd2,0xa2,0x00,0xf0,0xff,0xff]
          vpdpwsud ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vpdpwsud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x12,0xd2,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwsud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x12,0xd2,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwsud xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x12,0xd2,0x25,0x00,0x00,0x00,0x00]
          vpdpwsud xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpwsud xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x12,0xd2,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwsud xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpwsud xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x12,0xd2,0xa1,0xf0,0x07,0x00,0x00]
          vpdpwsud xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vpdpwsud xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x12,0xd2,0xa2,0x00,0xf8,0xff,0xff]
          vpdpwsud xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vpdpwsuds ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x16,0xd3,0xe4]
          vpdpwsuds ymm12, ymm13, ymm4

// CHECK: vpdpwsuds xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x12,0xd3,0xe4]
          vpdpwsuds xmm12, xmm13, xmm4

// CHECK: vpdpwsuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x16,0xd3,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwsuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x16,0xd3,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwsuds ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x16,0xd3,0x25,0x00,0x00,0x00,0x00]
          vpdpwsuds ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpwsuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x16,0xd3,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwsuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpwsuds ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x16,0xd3,0xa1,0xe0,0x0f,0x00,0x00]
          vpdpwsuds ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vpdpwsuds ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x16,0xd3,0xa2,0x00,0xf0,0xff,0xff]
          vpdpwsuds ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vpdpwsuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x12,0xd3,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwsuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwsuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x12,0xd3,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwsuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwsuds xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x12,0xd3,0x25,0x00,0x00,0x00,0x00]
          vpdpwsuds xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpwsuds xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x12,0xd3,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwsuds xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpwsuds xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x12,0xd3,0xa1,0xf0,0x07,0x00,0x00]
          vpdpwsuds xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vpdpwsuds xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x12,0xd3,0xa2,0x00,0xf8,0xff,0xff]
          vpdpwsuds xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vpdpwusd ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x15,0xd2,0xe4]
          vpdpwusd ymm12, ymm13, ymm4

// CHECK: vpdpwusd xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x11,0xd2,0xe4]
          vpdpwusd xmm12, xmm13, xmm4

// CHECK: vpdpwusd ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x15,0xd2,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusd ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwusd ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x15,0xd2,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusd ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwusd ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x15,0xd2,0x25,0x00,0x00,0x00,0x00]
          vpdpwusd ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpwusd ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x15,0xd2,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwusd ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpwusd ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x15,0xd2,0xa1,0xe0,0x0f,0x00,0x00]
          vpdpwusd ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vpdpwusd ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x15,0xd2,0xa2,0x00,0xf0,0xff,0xff]
          vpdpwusd ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vpdpwusd xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x11,0xd2,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusd xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwusd xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x11,0xd2,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusd xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwusd xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x11,0xd2,0x25,0x00,0x00,0x00,0x00]
          vpdpwusd xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpwusd xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x11,0xd2,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwusd xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpwusd xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x11,0xd2,0xa1,0xf0,0x07,0x00,0x00]
          vpdpwusd xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vpdpwusd xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x11,0xd2,0xa2,0x00,0xf8,0xff,0xff]
          vpdpwusd xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vpdpwusds ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x15,0xd3,0xe4]
          vpdpwusds ymm12, ymm13, ymm4

// CHECK: vpdpwusds xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x11,0xd3,0xe4]
          vpdpwusds xmm12, xmm13, xmm4

// CHECK: vpdpwusds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x15,0xd3,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwusds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x15,0xd3,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwusds ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x15,0xd3,0x25,0x00,0x00,0x00,0x00]
          vpdpwusds ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpwusds ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x15,0xd3,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwusds ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpwusds ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x15,0xd3,0xa1,0xe0,0x0f,0x00,0x00]
          vpdpwusds ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vpdpwusds ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x15,0xd3,0xa2,0x00,0xf0,0xff,0xff]
          vpdpwusds ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vpdpwusds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x11,0xd3,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwusds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwusds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x11,0xd3,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwusds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwusds xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x11,0xd3,0x25,0x00,0x00,0x00,0x00]
          vpdpwusds xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpwusds xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x11,0xd3,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwusds xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpwusds xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x11,0xd3,0xa1,0xf0,0x07,0x00,0x00]
          vpdpwusds xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vpdpwusds xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x11,0xd3,0xa2,0x00,0xf8,0xff,0xff]
          vpdpwusds xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vpdpwuud ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x14,0xd2,0xe4]
          vpdpwuud ymm12, ymm13, ymm4

// CHECK: vpdpwuud xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x10,0xd2,0xe4]
          vpdpwuud xmm12, xmm13, xmm4

// CHECK: vpdpwuud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x14,0xd2,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuud ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwuud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x14,0xd2,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuud ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwuud ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x14,0xd2,0x25,0x00,0x00,0x00,0x00]
          vpdpwuud ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpwuud ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x14,0xd2,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwuud ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpwuud ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x14,0xd2,0xa1,0xe0,0x0f,0x00,0x00]
          vpdpwuud ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vpdpwuud ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x14,0xd2,0xa2,0x00,0xf0,0xff,0xff]
          vpdpwuud ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vpdpwuud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x10,0xd2,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuud xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwuud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x10,0xd2,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuud xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwuud xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x10,0xd2,0x25,0x00,0x00,0x00,0x00]
          vpdpwuud xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpwuud xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x10,0xd2,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwuud xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpwuud xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x10,0xd2,0xa1,0xf0,0x07,0x00,0x00]
          vpdpwuud xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vpdpwuud xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x10,0xd2,0xa2,0x00,0xf8,0xff,0xff]
          vpdpwuud xmm12, xmm13, xmmword ptr [rdx - 2048]

// CHECK: vpdpwuuds ymm12, ymm13, ymm4
// CHECK: encoding: [0xc4,0x62,0x14,0xd3,0xe4]
          vpdpwuuds ymm12, ymm13, ymm4

// CHECK: vpdpwuuds xmm12, xmm13, xmm4
// CHECK: encoding: [0xc4,0x62,0x10,0xd3,0xe4]
          vpdpwuuds xmm12, xmm13, xmm4

// CHECK: vpdpwuuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x14,0xd3,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuuds ymm12, ymm13, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwuuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x14,0xd3,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuuds ymm12, ymm13, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwuuds ymm12, ymm13, ymmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x14,0xd3,0x25,0x00,0x00,0x00,0x00]
          vpdpwuuds ymm12, ymm13, ymmword ptr [rip]

// CHECK: vpdpwuuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0xc4,0x62,0x14,0xd3,0x24,0x6d,0x00,0xfc,0xff,0xff]
          vpdpwuuds ymm12, ymm13, ymmword ptr [2*rbp - 1024]

// CHECK: vpdpwuuds ymm12, ymm13, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0xc4,0x62,0x14,0xd3,0xa1,0xe0,0x0f,0x00,0x00]
          vpdpwuuds ymm12, ymm13, ymmword ptr [rcx + 4064]

// CHECK: vpdpwuuds ymm12, ymm13, ymmword ptr [rdx - 4096]
// CHECK: encoding: [0xc4,0x62,0x14,0xd3,0xa2,0x00,0xf0,0xff,0xff]
          vpdpwuuds ymm12, ymm13, ymmword ptr [rdx - 4096]

// CHECK: vpdpwuuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0x22,0x10,0xd3,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vpdpwuuds xmm12, xmm13, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vpdpwuuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0xc4,0x42,0x10,0xd3,0xa4,0x80,0x23,0x01,0x00,0x00]
          vpdpwuuds xmm12, xmm13, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vpdpwuuds xmm12, xmm13, xmmword ptr [rip]
// CHECK: encoding: [0xc4,0x62,0x10,0xd3,0x25,0x00,0x00,0x00,0x00]
          vpdpwuuds xmm12, xmm13, xmmword ptr [rip]

// CHECK: vpdpwuuds xmm12, xmm13, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0xc4,0x62,0x10,0xd3,0x24,0x6d,0x00,0xfe,0xff,0xff]
          vpdpwuuds xmm12, xmm13, xmmword ptr [2*rbp - 512]

// CHECK: vpdpwuuds xmm12, xmm13, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0xc4,0x62,0x10,0xd3,0xa1,0xf0,0x07,0x00,0x00]
          vpdpwuuds xmm12, xmm13, xmmword ptr [rcx + 2032]

// CHECK: vpdpwuuds xmm12, xmm13, xmmword ptr [rdx - 2048]
// CHECK: encoding: [0xc4,0x62,0x10,0xd3,0xa2,0x00,0xf8,0xff,0xff]
          vpdpwuuds xmm12, xmm13, xmmword ptr [rdx - 2048]

