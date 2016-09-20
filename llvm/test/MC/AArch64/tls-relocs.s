// RUN: llvm-mc -triple=aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s
// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj < %s -o - | \
// RUN:   llvm-readobj -r -t | FileCheck --check-prefix=CHECK-ELF %s

        // TLS local-dynamic forms
        movz x1, #:dtprel_g2:var
        movn x2, #:dtprel_g2:var
        movz x3, #:dtprel_g2:var
        movn x4, #:dtprel_g2:var

// CHECK: movz    x1, #:dtprel_g2:var     // encoding: [0bAAA00001,A,0b110AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_aarch64_movw
// CHECK: movn    x2, #:dtprel_g2:var     // encoding: [0bAAA00010,A,0b110AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_aarch64_movw
// CHECK: movz    x3, #:dtprel_g2:var     // encoding: [0bAAA00011,A,0b110AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_aarch64_movw
// CHECK: movn    x4, #:dtprel_g2:var     // encoding: [0bAAA00100,A,0b110AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_aarch64_movw

// CHECK-ELF:      Relocations [
// CHECK-ELF-NEXT:   Section {{.*}} .rela.text {
// CHECK-ELF-NEXT:     0x0 R_AARCH64_TLSLD_MOVW_DTPREL_G2 [[VARSYM:[^ ]+]]
// CHECK-ELF-NEXT:     0x4 R_AARCH64_TLSLD_MOVW_DTPREL_G2 [[VARSYM]]
// CHECK-ELF-NEXT:     0x8 R_AARCH64_TLSLD_MOVW_DTPREL_G2 [[VARSYM]]
// CHECK-ELF-NEXT:     0xC R_AARCH64_TLSLD_MOVW_DTPREL_G2 [[VARSYM]]


        movz x5, #:dtprel_g1:var
        movn x6, #:dtprel_g1:var
        movz w7, #:dtprel_g1:var
        movn w8, #:dtprel_g1:var

// CHECK: movz    x5, #:dtprel_g1:var     // encoding: [0bAAA00101,A,0b101AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movn    x6, #:dtprel_g1:var     // encoding: [0bAAA00110,A,0b101AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movz    w7, #:dtprel_g1:var     // encoding: [0bAAA00111,A,0b101AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movn    w8, #:dtprel_g1:var     // encoding: [0bAAA01000,A,0b101AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x10 R_AARCH64_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0x14 R_AARCH64_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0x18 R_AARCH64_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0x1C R_AARCH64_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]


        movk x9, #:dtprel_g1_nc:var
        movk w10, #:dtprel_g1_nc:var

// CHECK: movk    x9, #:dtprel_g1_nc:var  // encoding: [0bAAA01001,A,0b101AAAAA,0xf2]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g1_nc:var, kind: fixup_aarch64_movw
// CHECK: movk    w10, #:dtprel_g1_nc:var // encoding: [0bAAA01010,A,0b101AAAAA,0x72]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g1_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x20 R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x24 R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC [[VARSYM]]


        movz x11, #:dtprel_g0:var
        movn x12, #:dtprel_g0:var
        movz w13, #:dtprel_g0:var
        movn w14, #:dtprel_g0:var

// CHECK: movz    x11, #:dtprel_g0:var    // encoding: [0bAAA01011,A,0b100AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw
// CHECK: movn    x12, #:dtprel_g0:var    // encoding: [0bAAA01100,A,0b100AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw
// CHECK: movz    w13, #:dtprel_g0:var    // encoding: [0bAAA01101,A,0b100AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw
// CHECK: movn    w14, #:dtprel_g0:var    // encoding: [0bAAA01110,A,0b100AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x28 R_AARCH64_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]
// CHECK-ELF-NEXT:     0x2C R_AARCH64_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]
// CHECK-ELF-NEXT:     0x30 R_AARCH64_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]
// CHECK-ELF-NEXT:     0x34 R_AARCH64_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]


        movk x15, #:dtprel_g0_nc:var
        movk w16, #:dtprel_g0_nc:var

// CHECK: movk    x15, #:dtprel_g0_nc:var // encoding: [0bAAA01111,A,0b100AAAAA,0xf2]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g0_nc:var, kind: fixup_aarch64_movw
// CHECK: movk    w16, #:dtprel_g0_nc:var // encoding: [0bAAA10000,A,0b100AAAAA,0x72]
// CHECK:                                 //   fixup A - offset: 0, value: :dtprel_g0_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x38 R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x3C R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC [[VARSYM]]


        add x17, x18, #:dtprel_hi12:var, lsl #12
        add w19, w20, #:dtprel_hi12:var, lsl #12

// CHECK: add    x17, x18, :dtprel_hi12:var, lsl #12 // encoding: [0x51,0bAAAAAA10,0b01AAAAAA,0x91]
// CHECK:                                            //   fixup A - offset: 0, value: :dtprel_hi12:var, kind: fixup_aarch64_add_imm12
// CHECK: add    w19, w20, :dtprel_hi12:var, lsl #12 // encoding: [0x93,0bAAAAAA10,0b01AAAAAA,0x11]
// CHECK:                                            //   fixup A - offset: 0, value: :dtprel_hi12:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-NEXT:     0x40 R_AARCH64_TLSLD_ADD_DTPREL_HI12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x44 R_AARCH64_TLSLD_ADD_DTPREL_HI12 [[VARSYM]]


        add x21, x22, #:dtprel_lo12:var
        add w23, w24, #:dtprel_lo12:var

// CHECK: add    x21, x22, :dtprel_lo12:var // encoding: [0xd5,0bAAAAAA10,0b00AAAAAA,0x91]
// CHECK:                                   //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_add_imm12
// CHECK: add    w23, w24, :dtprel_lo12:var // encoding: [0x17,0bAAAAAA11,0b00AAAAAA,0x11]
// CHECK:                                   //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-NEXT:     0x48 R_AARCH64_TLSLD_ADD_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x4C R_AARCH64_TLSLD_ADD_DTPREL_LO12 [[VARSYM]]


        add x25, x26, #:dtprel_lo12_nc:var
        add w27, w28, #:dtprel_lo12_nc:var

// CHECK: add    x25, x26, :dtprel_lo12_nc:var // encoding: [0x59,0bAAAAAA11,0b00AAAAAA,0x91]
// CHECK:                                      //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_add_imm12
// CHECK: add    w27, w28, :dtprel_lo12_nc:var // encoding: [0x9b,0bAAAAAA11,0b00AAAAAA,0x11]
// CHECK:                                      //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-NEXT:     0x50 R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x54 R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC [[VARSYM]]


        ldrb w29, [x30, #:dtprel_lo12:var]
        ldrsb x29, [x28, #:dtprel_lo12_nc:var]

// CHECK: ldrb    w29, [x30, :dtprel_lo12:var] // encoding: [0xdd,0bAAAAAA11,0b01AAAAAA,0x39]
// CHECK:                                      //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale1
// CHECK: ldrsb    x29, [x28, :dtprel_lo12_nc:var] // encoding: [0x9d,0bAAAAAA11,0b10AAAAAA,0x39]
// CHECK:                                          //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale1

// CHECK-ELF-NEXT:     0x58 R_AARCH64_TLSLD_LDST8_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x5C R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC [[VARSYM]]


        strh w27, [x26, #:dtprel_lo12:var]
        ldrsh x25, [x24, #:dtprel_lo12_nc:var]

// CHECK: strh    w27, [x26, :dtprel_lo12:var] // encoding: [0x5b,0bAAAAAA11,0b00AAAAAA,0x79]
// CHECK:                                      //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale2
// CHECK: ldrsh    x25, [x24, :dtprel_lo12_nc:var] // encoding: [0x19,0bAAAAAA11,0b10AAAAAA,0x79]
// CHECK:                                          //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale2

// CHECK-ELF-NEXT:     0x60 R_AARCH64_TLSLD_LDST16_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x64 R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC [[VARSYM]]


        ldr w23, [x22, #:dtprel_lo12:var]
        ldrsw x21, [x20, #:dtprel_lo12_nc:var]

// CHECK: ldr    w23, [x22, :dtprel_lo12:var] // encoding: [0xd7,0bAAAAAA10,0b01AAAAAA,0xb9]
// CHECK:                                     //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK: ldrsw    x21, [x20, :dtprel_lo12_nc:var] // encoding: [0x95,0bAAAAAA10,0b10AAAAAA,0xb9]
// CHECK:                                          //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale4

// CHECK-ELF-NEXT:     0x68 R_AARCH64_TLSLD_LDST32_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x6C R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC [[VARSYM]]


        ldr x19, [x18, #:dtprel_lo12:var]
        str x17, [x16, #:dtprel_lo12_nc:var]

// CHECK: ldr    x19, [x18, :dtprel_lo12:var] // encoding: [0x53,0bAAAAAA10,0b01AAAAAA,0xf9]
// CHECK:                                     //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK: str    x17, [x16, :dtprel_lo12_nc:var] // encoding: [0x11,0bAAAAAA10,0b00AAAAAA,0xf9]
// CHECK:                                        //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale8

// CHECK-ELF-NEXT:     0x70 R_AARCH64_TLSLD_LDST64_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x74 R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC [[VARSYM]]


        // TLS initial-exec forms
        movz x15, #:gottprel_g1:var
        movz w14, #:gottprel_g1:var

// CHECK: movz    x15, #:gottprel_g1:var  // encoding: [0bAAA01111,A,0b101AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :gottprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movz    w14, #:gottprel_g1:var  // encoding: [0bAAA01110,A,0b101AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :gottprel_g1:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x78 R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0x7C R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 [[VARSYM]]


        movk x13, #:gottprel_g0_nc:var
        movk w12, #:gottprel_g0_nc:var

// CHECK: movk    x13, #:gottprel_g0_nc:var // encoding: [0bAAA01101,A,0b100AAAAA,0xf2]
// CHECK:                                   //   fixup A - offset: 0, value: :gottprel_g0_nc:var, kind: fixup_aarch64_movw
// CHECK: movk    w12, #:gottprel_g0_nc:var // encoding: [0bAAA01100,A,0b100AAAAA,0x72]
// CHECK:                                   //   fixup A - offset: 0, value: :gottprel_g0_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x80 R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x84 R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC [[VARSYM]]


        adrp x11, :gottprel:var
        ldr x10, [x0, #:gottprel_lo12:var]
        ldr x9, :gottprel:var

// CHECK: adrp    x11, :gottprel:var      // encoding: [0x0b'A',A,A,0x90'A']
// CHECK:                                 //   fixup A - offset: 0, value: :gottprel:var, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK: ldr    x10, [x0, :gottprel_lo12:var] // encoding: [0x0a,0bAAAAAA00,0b01AAAAAA,0xf9]
// CHECK:                                      //   fixup A - offset: 0, value: :gottprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK: ldr    x9, :gottprel:var       // encoding: [0bAAA01001,A,A,0x58]
// CHECK:                                //   fixup A - offset: 0, value: :gottprel:var, kind: fixup_aarch64_ldr_pcrel_imm19

// CHECK-ELF-NEXT:     0x88 R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 [[VARSYM]]
// CHECK-ELF-NEXT:     0x8C R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x90 R_AARCH64_TLSIE_LD_GOTTPREL_PREL19 [[VARSYM]]


        // TLS local-exec forms
        movz x3, #:tprel_g2:var
        movn x4, #:tprel_g2:var

// CHECK: movz    x3, #:tprel_g2:var      // encoding: [0bAAA00011,A,0b110AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g2:var, kind: fixup_aarch64_movw
// CHECK: movn    x4, #:tprel_g2:var      // encoding: [0bAAA00100,A,0b110AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g2:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x94 R_AARCH64_TLSLE_MOVW_TPREL_G2 [[VARSYM]]
// CHECK-ELF-NEXT:     0x98 R_AARCH64_TLSLE_MOVW_TPREL_G2 [[VARSYM]]


        movz x5, #:tprel_g1:var
        movn x6, #:tprel_g1:var
        movz w7, #:tprel_g1:var
        movn w8, #:tprel_g1:var

// CHECK: movz    x5, #:tprel_g1:var      // encoding: [0bAAA00101,A,0b101AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movn    x6, #:tprel_g1:var      // encoding: [0bAAA00110,A,0b101AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movz    w7, #:tprel_g1:var      // encoding: [0bAAA00111,A,0b101AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw
// CHECK: movn    w8, #:tprel_g1:var      // encoding: [0bAAA01000,A,0b101AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0x9C R_AARCH64_TLSLE_MOVW_TPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0xA0 R_AARCH64_TLSLE_MOVW_TPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0xA4 R_AARCH64_TLSLE_MOVW_TPREL_G1 [[VARSYM]]
// CHECK-ELF-NEXT:     0xA8 R_AARCH64_TLSLE_MOVW_TPREL_G1 [[VARSYM]]


        movk x9, #:tprel_g1_nc:var
        movk w10, #:tprel_g1_nc:var

// CHECK: movk    x9, #:tprel_g1_nc:var   // encoding: [0bAAA01001,A,0b101AAAAA,0xf2]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g1_nc:var, kind: fixup_aarch64_movw
// CHECK: movk    w10, #:tprel_g1_nc:var  // encoding: [0bAAA01010,A,0b101AAAAA,0x72]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g1_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0xAC R_AARCH64_TLSLE_MOVW_TPREL_G1_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0xB0 R_AARCH64_TLSLE_MOVW_TPREL_G1_NC [[VARSYM]]


        movz x11, #:tprel_g0:var
        movn x12, #:tprel_g0:var
        movz w13, #:tprel_g0:var
        movn w14, #:tprel_g0:var

// CHECK: movz    x11, #:tprel_g0:var     // encoding: [0bAAA01011,A,0b100AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw
// CHECK: movn    x12, #:tprel_g0:var     // encoding: [0bAAA01100,A,0b100AAAAA,0x92]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw
// CHECK: movz    w13, #:tprel_g0:var     // encoding: [0bAAA01101,A,0b100AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw
// CHECK: movn    w14, #:tprel_g0:var     // encoding: [0bAAA01110,A,0b100AAAAA,0x12]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0xB4 R_AARCH64_TLSLE_MOVW_TPREL_G0 [[VARSYM]]
// CHECK-ELF-NEXT:     0xB8 R_AARCH64_TLSLE_MOVW_TPREL_G0 [[VARSYM]]
// CHECK-ELF-NEXT:     0xBC R_AARCH64_TLSLE_MOVW_TPREL_G0 [[VARSYM]]
// CHECK-ELF-NEXT:     0xC0 R_AARCH64_TLSLE_MOVW_TPREL_G0 [[VARSYM]]


        movk x15, #:tprel_g0_nc:var
        movk w16, #:tprel_g0_nc:var

// CHECK: movk    x15, #:tprel_g0_nc:var  // encoding: [0bAAA01111,A,0b100AAAAA,0xf2]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g0_nc:var, kind: fixup_aarch64_movw
// CHECK: movk    w16, #:tprel_g0_nc:var  // encoding: [0bAAA10000,A,0b100AAAAA,0x72]
// CHECK:                                 //   fixup A - offset: 0, value: :tprel_g0_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-NEXT:     0xC4 R_AARCH64_TLSLE_MOVW_TPREL_G0_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0xC8 R_AARCH64_TLSLE_MOVW_TPREL_G0_NC [[VARSYM]]


        add x17, x18, #:tprel_hi12:var, lsl #12
        add w19, w20, #:tprel_hi12:var, lsl #12

// CHECK: add    x17, x18, :tprel_hi12:var, lsl #12 // encoding: [0x51,0bAAAAAA10,0b01AAAAAA,0x91]
// CHECK:                                           //   fixup A - offset: 0, value: :tprel_hi12:var, kind: fixup_aarch64_add_imm12
// CHECK: add    w19, w20, :tprel_hi12:var, lsl #12 // encoding: [0x93,0bAAAAAA10,0b01AAAAAA,0x11]
// CHECK:                                           //   fixup A - offset: 0, value: :tprel_hi12:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-NEXT:     0xCC R_AARCH64_TLSLE_ADD_TPREL_HI12 [[VARSYM]]
// CHECK-ELF-NEXT:     0xD0 R_AARCH64_TLSLE_ADD_TPREL_HI12 [[VARSYM]]


        add x21, x22, #:tprel_lo12:var
        add w23, w24, #:tprel_lo12:var

// CHECK: add    x21, x22, :tprel_lo12:var // encoding: [0xd5,0bAAAAAA10,0b00AAAAAA,0x91]
// CHECK:                                  //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_add_imm12
// CHECK: add    w23, w24, :tprel_lo12:var // encoding: [0x17,0bAAAAAA11,0b00AAAAAA,0x11]
// CHECK:                                  //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-NEXT:     0xD4 R_AARCH64_TLSLE_ADD_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0xD8 R_AARCH64_TLSLE_ADD_TPREL_LO12 [[VARSYM]]


        add x25, x26, #:tprel_lo12_nc:var
        add w27, w28, #:tprel_lo12_nc:var

// CHECK: add    x25, x26, :tprel_lo12_nc:var // encoding: [0x59,0bAAAAAA11,0b00AAAAAA,0x91]
// CHECK:                                     //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_add_imm12
// CHECK: add    w27, w28, :tprel_lo12_nc:var // encoding: [0x9b,0bAAAAAA11,0b00AAAAAA,0x11]
// CHECK:                                     //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-NEXT:     0xDC R_AARCH64_TLSLE_ADD_TPREL_LO12_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0xE0 R_AARCH64_TLSLE_ADD_TPREL_LO12_NC [[VARSYM]]


        ldrb w29, [x30, #:tprel_lo12:var]
        ldrsb x29, [x28, #:tprel_lo12_nc:var]

// CHECK: ldrb    w29, [x30, :tprel_lo12:var] // encoding: [0xdd,0bAAAAAA11,0b01AAAAAA,0x39]
// CHECK:                                     //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale1
// CHECK: ldrsb    x29, [x28, :tprel_lo12_nc:var] // encoding: [0x9d,0bAAAAAA11,0b10AAAAAA,0x39]
// CHECK:                                     //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale1

// CHECK-ELF-NEXT:     0xE4 R_AARCH64_TLSLE_LDST8_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0xE8 R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC [[VARSYM]]


        strh w27, [x26, #:tprel_lo12:var]
        ldrsh x25, [x24, #:tprel_lo12_nc:var]

// CHECK: strh    w27, [x26, :tprel_lo12:var] // encoding: [0x5b,0bAAAAAA11,0b00AAAAAA,0x79]
// CHECK:                                     //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale2
// CHECK: ldrsh    x25, [x24, :tprel_lo12_nc:var] // encoding: [0x19,0bAAAAAA11,0b10AAAAAA,0x79]
// CHECK:                                         //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale2

// CHECK-ELF-NEXT:     0xEC R_AARCH64_TLSLE_LDST16_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0xF0 R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC [[VARSYM]]


        ldr w23, [x22, #:tprel_lo12:var]
        ldrsw x21, [x20, #:tprel_lo12_nc:var]

// CHECK: ldr    w23, [x22, :tprel_lo12:var] // encoding: [0xd7,0bAAAAAA10,0b01AAAAAA,0xb9]
// CHECK:                                    //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK: ldrsw    x21, [x20, :tprel_lo12_nc:var] // encoding: [0x95,0bAAAAAA10,0b10AAAAAA,0xb9]
// CHECK:                                         //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale4

// CHECK-ELF-NEXT:     0xF4 R_AARCH64_TLSLE_LDST32_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0xF8 R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC [[VARSYM]]

        ldr x19, [x18, #:tprel_lo12:var]
        str x17, [x16, #:tprel_lo12_nc:var]

// CHECK: ldr    x19, [x18, :tprel_lo12:var] // encoding: [0x53,0bAAAAAA10,0b01AAAAAA,0xf9]
// CHECK:                                    //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK: str    x17, [x16, :tprel_lo12_nc:var] // encoding: [0x11,0bAAAAAA10,0b00AAAAAA,0xf9]
// CHECK:                                       //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale8

// CHECK-ELF-NEXT:     0xFC  R_AARCH64_TLSLE_LDST64_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x100 R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC [[VARSYM]]

        // TLS descriptor forms
        adrp x8, :tlsdesc:var
        ldr x7, [x6, :tlsdesc_lo12:var]
        add x5, x4, #:tlsdesc_lo12:var
        .tlsdesccall var
        blr x3


// CHECK: adrp    x8, :tlsdesc:var        // encoding: [0x08'A',A,A,0x90'A']
// CHECK:                                 //   fixup A - offset: 0, value: :tlsdesc:var, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK: ldr    x7, [x6, :tlsdesc_lo12:var] // encoding: [0xc7,0bAAAAAA00,0b01AAAAAA,0xf9]
// CHECK:                                    //   fixup A - offset: 0, value: :tlsdesc_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK: add    x5, x4, :tlsdesc_lo12:var // encoding: [0x85,0bAAAAAA00,0b00AAAAAA,0x91]
// CHECK:                                  //   fixup A - offset: 0, value: :tlsdesc_lo12:var, kind: fixup_aarch64_add_imm12
// CHECK: .tlsdesccall var                // encoding: []
// CHECK:                                 //   fixup A - offset: 0, value: var, kind: fixup_aarch64_tlsdesc_call
// CHECK: blr    x3                      // encoding: [0x60,0x00,0x3f,0xd6]

// CHECK-ELF-NEXT:     0x104 R_AARCH64_TLSDESC_ADR_PAGE21 [[VARSYM]]
// CHECK-ELF-NEXT:     0x108 R_AARCH64_TLSDESC_LD64_LO12_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x10C R_AARCH64_TLSDESC_ADD_LO12_NC [[VARSYM]]
// CHECK-ELF-NEXT:     0x110 R_AARCH64_TLSDESC_CALL [[VARSYM]]


// Make sure symbol 5 has type STT_TLS:

// CHECK-ELF:      Symbols [
// CHECK-ELF:        Symbol {
// CHECK-ELF:          Name: var
// CHECK-ELF-NEXT:     Value:
// CHECK-ELF-NEXT:     Size:
// CHECK-ELF-NEXT:     Binding: Global
// CHECK-ELF-NEXT:     Type: TLS
