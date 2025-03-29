// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vaesenc %ymm3, %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdc,0xcb]
          vaesenc %ymm3, %ymm2, %ymm1

// CHECK: vaesenclast %ymm3, %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdd,0xcb]
          vaesenclast %ymm3, %ymm2, %ymm1

// CHECK: vaesdec %ymm3, %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xde,0xcb]
          vaesdec %ymm3, %ymm2, %ymm1

// CHECK: vaesdeclast %ymm3, %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdf,0xcb]
          vaesdeclast %ymm3, %ymm2, %ymm1

// CHECK: vaesenc  (%rcx), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdc,0x09]
          vaesenc  (%rcx), %ymm2, %ymm1

// CHECK: vaesenc  -4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdc,0x4c,0x24,0xfc]
          vaesenc  -4(%rsp), %ymm2, %ymm1

// CHECK: vaesenc  4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdc,0x4c,0x24,0x04]
          vaesenc  4(%rsp), %ymm2, %ymm1

// CHECK: vaesenc  268435456(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdc,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vaesenc  268435456(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesenc  -536870912(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdc,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vaesenc  -536870912(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesenc  -536870910(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdc,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vaesenc  -536870910(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesenclast  (%rcx), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdd,0x09]
          vaesenclast  (%rcx), %ymm2, %ymm1

// CHECK: vaesenclast  -4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdd,0x4c,0x24,0xfc]
          vaesenclast  -4(%rsp), %ymm2, %ymm1

// CHECK: vaesenclast  4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdd,0x4c,0x24,0x04]
          vaesenclast  4(%rsp), %ymm2, %ymm1

// CHECK: vaesenclast  268435456(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdd,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vaesenclast  268435456(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesenclast  -536870912(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdd,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vaesenclast  -536870912(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesenclast  -536870910(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdd,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vaesenclast  -536870910(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesdec  (%rcx), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xde,0x09]
          vaesdec  (%rcx), %ymm2, %ymm1

// CHECK: vaesdec  -4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xde,0x4c,0x24,0xfc]
          vaesdec  -4(%rsp), %ymm2, %ymm1

// CHECK: vaesdec  4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xde,0x4c,0x24,0x04]
          vaesdec  4(%rsp), %ymm2, %ymm1

// CHECK: vaesdec  268435456(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xde,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vaesdec  268435456(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesdec  -536870912(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xde,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vaesdec  -536870912(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesdec  -536870910(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xde,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vaesdec  -536870910(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesdeclast  (%rcx), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdf,0x09]
          vaesdeclast  (%rcx), %ymm2, %ymm1

// CHECK: vaesdeclast  -4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdf,0x4c,0x24,0xfc]
          vaesdeclast  -4(%rsp), %ymm2, %ymm1

// CHECK: vaesdeclast  4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x6d,0xdf,0x4c,0x24,0x04]
          vaesdeclast  4(%rsp), %ymm2, %ymm1

// CHECK: vaesdeclast  268435456(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vaesdeclast  268435456(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesdeclast  -536870912(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vaesdeclast  -536870912(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vaesdeclast  -536870910(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x6d,0xdf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vaesdeclast  -536870910(%rcx,%r14,8), %ymm2, %ymm1

