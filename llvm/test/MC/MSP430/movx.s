// Test the encoding of MOVX instructions for the MSP430X CPU.

// RUN: llvm-mc -triple msp430 -mcpu=msp430x -show-encoding < %s | FileCheck %s
movx_a:
  movx.a @r13, 0(r12)
  movx.a @r13, 2(r12)
  movx.a @r13, 0x12345(r12)
  movx.a @r13, 0xfffff(r12)
  movx.a @r13, -2(r12)
  movx.a @r13, -0x80000(r12)
  movx.a @r4, &0x0512
  movx.a @r4, &0x10512

// CHECK-LABEL: movx_a:
// CHECK-NEXT: movx.a	@r13, 0(r12)                    ; encoding: [0x00,0x18,0xec,0x4d,0x00,0x00]
// CHECK-NEXT: movx.a	@r13, 2(r12)                    ; encoding: [0x00,0x18,0xec,0x4d,0x02,0x00]
// CHECK-NEXT: movx.a	@r13, 74565(r12)                ; encoding: [0x01,0x18,0xec,0x4d,0x45,0x23]
// CHECK-NEXT: movx.a	@r13, 1048575(r12)              ; encoding: [0x0f,0x18,0xec,0x4d,0xff,0xff]
// CHECK-NEXT: movx.a	@r13, -2(r12)                   ; encoding: [0x0f,0x18,0xec,0x4d,0xfe,0xff]
// CHECK-NEXT: movx.a	@r13, -524288(r12)              ; encoding: [0x08,0x18,0xec,0x4d,0x00,0x00]
// CHECK-NEXT: movx.a	@r4, &1298                      ; encoding: [0x00,0x18,0xe2,0x44,0x12,0x05]
// CHECK-NEXT: movx.a	@r4, &66834                     ; encoding: [0x01,0x18,0xe2,0x44,0x12,0x05]

movx_w:
  movx @r13, 0(r12)
  movx.w @r13, 0(r12)
  movx @r13, 0x12345(r12)
  movx @r4, &0x0512

// CHECK-LABEL: movx_w:
// CHECK-NEXT: movx	@r13, 0(r12)                    ; encoding: [0x40,0x18,0xac,0x4d,0x00,0x00]
// CHECK-NEXT: movx	@r13, 0(r12)                    ; encoding: [0x40,0x18,0xac,0x4d,0x00,0x00]
// CHECK-NEXT: movx	@r13, 74565(r12)                ; encoding: [0x41,0x18,0xac,0x4d,0x45,0x23]
// CHECK-NEXT: movx	@r4, &1298                      ; encoding: [0x40,0x18,0xa2,0x44,0x12,0x05]
