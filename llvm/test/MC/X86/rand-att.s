// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdrandw %ax
// CHECK: encoding: [0x66,0x0f,0xc7,0xf0]
          rdrand %ax

// CHECK: rdrandl %eax
// CHECK: encoding: [0x0f,0xc7,0xf0]
          rdrand %eax

// CHECK: rdrandq %rax
// CHECK: encoding: [0x48,0x0f,0xc7,0xf0]
          rdrand %rax

// CHECK: rdrandw %r11w
// CHECK: encoding: [0x66,0x41,0x0f,0xc7,0xf3]
          rdrand %r11w

// CHECK: rdrandl %r11d
// CHECK: encoding: [0x41,0x0f,0xc7,0xf3]
          rdrand %r11d

// CHECK: rdrandq %r11
// CHECK: encoding: [0x49,0x0f,0xc7,0xf3]
          rdrand %r11

// CHECK: rdseedw %ax
// CHECK: encoding: [0x66,0x0f,0xc7,0xf8]
          rdseed %ax

// CHECK: rdseedl %eax
// CHECK: encoding: [0x0f,0xc7,0xf8]
          rdseed %eax

// CHECK: rdseedq %rax
// CHECK: encoding: [0x48,0x0f,0xc7,0xf8]
          rdseed %rax

// CHECK: rdseedw %r11w
// CHECK: encoding: [0x66,0x41,0x0f,0xc7,0xfb]
          rdseed %r11w

// CHECK: rdseedl %r11d
// CHECK: encoding: [0x41,0x0f,0xc7,0xfb]
          rdseed %r11d

// CHECK: rdseedq %r11
// CHECK: encoding: [0x49,0x0f,0xc7,0xfb]
          rdseed %r11
