// RUN: llvm-mc -triple=x86_64-unknown-unknown -mattr=+avxifma --show-encoding < %s  | FileCheck %s

// CHECK: {vex} vpmadd52huq %ymm14, %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x42,0x95,0xb5,0xe6]
     {vex} vpmadd52huq %ymm14, %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq %xmm14, %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x42,0x91,0xb5,0xe6]
     {vex} vpmadd52huq %xmm14, %xmm13, %xmm12

// CHECK: {vex} vpmadd52huq  268435456(%rbp,%r14,8), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x22,0x95,0xb5,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq  268435456(%rbp,%r14,8), %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq  291(%r8,%rax,4), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x42,0x95,0xb5,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq  291(%r8,%rax,4), %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq  (%rip), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52huq  (%rip), %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq  -1024(,%rbp,2), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0x24,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52huq  -1024(,%rbp,2), %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq  4064(%rcx), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0xa1,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52huq  4064(%rcx), %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq  -4096(%rdx), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb5,0xa2,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52huq  -4096(%rdx), %ymm13, %ymm12

// CHECK: {vex} vpmadd52huq  268435456(%rbp,%r14,8), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x22,0x91,0xb5,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52huq  268435456(%rbp,%r14,8), %xmm13, %xmm12

// CHECK: {vex} vpmadd52huq  291(%r8,%rax,4), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x42,0x91,0xb5,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52huq  291(%r8,%rax,4), %xmm13, %xmm12

// CHECK: {vex} vpmadd52huq  (%rip), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52huq  (%rip), %xmm13, %xmm12

// CHECK: {vex} vpmadd52huq  -512(,%rbp,2), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0x24,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52huq  -512(,%rbp,2), %xmm13, %xmm12

// CHECK: {vex} vpmadd52huq  2032(%rcx), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0xa1,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52huq  2032(%rcx), %xmm13, %xmm12

// CHECK: {vex} vpmadd52huq  -2048(%rdx), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb5,0xa2,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52huq  -2048(%rdx), %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq %ymm14, %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x42,0x95,0xb4,0xe6]
     {vex} vpmadd52luq %ymm14, %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq %xmm14, %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x42,0x91,0xb4,0xe6]
     {vex} vpmadd52luq %xmm14, %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq  268435456(%rbp,%r14,8), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x22,0x95,0xb4,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq  268435456(%rbp,%r14,8), %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq  291(%r8,%rax,4), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x42,0x95,0xb4,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq  291(%r8,%rax,4), %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq  (%rip), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52luq  (%rip), %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq  -1024(,%rbp,2), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0x24,0x6d,0x00,0xfc,0xff,0xff]
     {vex} vpmadd52luq  -1024(,%rbp,2), %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq  4064(%rcx), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0xa1,0xe0,0x0f,0x00,0x00]
     {vex} vpmadd52luq  4064(%rcx), %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq  -4096(%rdx), %ymm13, %ymm12
// CHECK: encoding: [0xc4,0x62,0x95,0xb4,0xa2,0x00,0xf0,0xff,0xff]
     {vex} vpmadd52luq  -4096(%rdx), %ymm13, %ymm12

// CHECK: {vex} vpmadd52luq  268435456(%rbp,%r14,8), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x22,0x91,0xb4,0xa4,0xf5,0x00,0x00,0x00,0x10]
     {vex} vpmadd52luq  268435456(%rbp,%r14,8), %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq  291(%r8,%rax,4), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x42,0x91,0xb4,0xa4,0x80,0x23,0x01,0x00,0x00]
     {vex} vpmadd52luq  291(%r8,%rax,4), %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq  (%rip), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0x25,0x00,0x00,0x00,0x00]
     {vex} vpmadd52luq  (%rip), %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq  -512(,%rbp,2), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0x24,0x6d,0x00,0xfe,0xff,0xff]
     {vex} vpmadd52luq  -512(,%rbp,2), %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq  2032(%rcx), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0xa1,0xf0,0x07,0x00,0x00]
     {vex} vpmadd52luq  2032(%rcx), %xmm13, %xmm12

// CHECK: {vex} vpmadd52luq  -2048(%rdx), %xmm13, %xmm12
// CHECK: encoding: [0xc4,0x62,0x91,0xb4,0xa2,0x00,0xf8,0xff,0xff]
     {vex} vpmadd52luq  -2048(%rdx), %xmm13, %xmm12

