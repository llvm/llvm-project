// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8a,+fp-armv8 %s | FileCheck %s
  .globl _func

// Check that the assembler can handle the documented syntax from the ARM ARM.
// For complex constructs like shifter operands, check more thoroughly for them
// once then spot check that following instructions accept the form generally.
// This gives us good coverage while keeping the overall size of the test
// more reasonable.


_func:
// CHECK: _func

//------------------------------------------------------------------------------
// Add/sub (extended register)
//------------------------------------------------------------------------------
        // Basic extends 64-bit ops
        add x2, x4, w5, uxtb
        add x20, sp, w19, uxth
        add x12, x1, w20, uxtw
        add x20, x3, x13, uxtx
        add x17, x25, w20, sxtb
        add x18, x13, w19, sxth
        add sp, x2, w3, sxtw
        add x3, x5, x9, sxtx
// CHECK: add      x2, x4, w5, uxtb           // encoding: [0x82,0x00,0x25,0x8b]
// CHECK: add      x20, sp, w19, uxth         // encoding: [0xf4,0x23,0x33,0x8b]
// CHECK: add      x12, x1, w20, uxtw         // encoding: [0x2c,0x40,0x34,0x8b]
// CHECK: add      x20, x3, x13, uxtx         // encoding: [0x74,0x60,0x2d,0x8b]
// CHECK: add      x17, x25, w20, sxtb        // encoding: [0x31,0x83,0x34,0x8b]
// CHECK: add      x18, x13, w19, sxth        // encoding: [0xb2,0xa1,0x33,0x8b]
// CHECK: add      sp, x2, w3, sxtw           // encoding: [0x5f,0xc0,0x23,0x8b]
// CHECK: add      x3, x5, x9, sxtx           // encoding: [0xa3,0xe0,0x29,0x8b]

        // Basic extends, 32-bit ops
        add w2, w5, w7, uxtb
        add w21, w15, w17, uxth
        add w30, w29, wzr, uxtw
        add w19, w17, w1, uxtx  // Goodness knows what this means
        add w2, w5, w1, sxtb
        add w26, w17, w19, sxth
        add w0, w2, w3, sxtw
        add w2, w3, w5, sxtx
// CHECK: add      w2, w5, w7, uxtb           // encoding: [0xa2,0x00,0x27,0x0b]
// CHECK: add      w21, w15, w17, uxth        // encoding: [0xf5,0x21,0x31,0x0b]
// CHECK: add      w30, w29, wzr, uxtw        // encoding: [0xbe,0x43,0x3f,0x0b]
// CHECK: add      w19, w17, w1, uxtx         // encoding: [0x33,0x62,0x21,0x0b]
// CHECK: add      w2, w5, w1, sxtb           // encoding: [0xa2,0x80,0x21,0x0b]
// CHECK: add      w26, w17, w19, sxth        // encoding: [0x3a,0xa2,0x33,0x0b]
// CHECK: add      w0, w2, w3, sxtw           // encoding: [0x40,0xc0,0x23,0x0b]
// CHECK: add      w2, w3, w5, sxtx           // encoding: [0x62,0xe0,0x25,0x0b]

        // Nonzero shift amounts
        add x2, x3, w5, sxtb #0
        add x7, x11, w13, uxth #4
        add w17, w19, w23, uxtw #2
        add w29, w23, w17, uxtx #1
// CHECK: add      x2, x3, w5, sxtb           // encoding: [0x62,0x80,0x25,0x8b]
// CHECK: add      x7, x11, w13, uxth #4      // encoding: [0x67,0x31,0x2d,0x8b]
// CHECK: add      w17, w19, w23, uxtw #2     // encoding: [0x71,0x4a,0x37,0x0b]
// CHECK: add      w29, w23, w17, uxtx #1     // encoding: [0xfd,0x66,0x31,0x0b]

        // Sub
        sub x2, x4, w5, uxtb #2
        sub x20, sp, w19, uxth #4
        sub x12, x1, w20, uxtw
        sub x20, x3, x13, uxtx #0
        sub x17, x25, w20, sxtb
        sub x18, x13, w19, sxth
        sub sp, x2, w3, sxtw
        sub x3, x5, x9, sxtx
// CHECK: sub      x2, x4, w5, uxtb #2        // encoding: [0x82,0x08,0x25,0xcb]
// CHECK: sub      x20, sp, w19, uxth #4      // encoding: [0xf4,0x33,0x33,0xcb]
// CHECK: sub      x12, x1, w20, uxtw         // encoding: [0x2c,0x40,0x34,0xcb]
// CHECK: sub      x20, x3, x13, uxtx         // encoding: [0x74,0x60,0x2d,0xcb]
// CHECK: sub      x17, x25, w20, sxtb        // encoding: [0x31,0x83,0x34,0xcb]
// CHECK: sub      x18, x13, w19, sxth        // encoding: [0xb2,0xa1,0x33,0xcb]
// CHECK: sub      sp, x2, w3, sxtw           // encoding: [0x5f,0xc0,0x23,0xcb]
// CHECK: sub      x3, x5, x9, sxtx           // encoding: [0xa3,0xe0,0x29,0xcb]

        sub w2, w5, w7, uxtb
        sub w21, w15, w17, uxth
        sub w30, w29, wzr, uxtw
        sub w19, w17, w1, uxtx  // Goodness knows what this means
        sub w2, w5, w1, sxtb
        sub w26, wsp, w19, sxth
        sub wsp, w2, w3, sxtw
        sub w2, w3, w5, sxtx
// CHECK: sub      w2, w5, w7, uxtb           // encoding: [0xa2,0x00,0x27,0x4b]
// CHECK: sub      w21, w15, w17, uxth        // encoding: [0xf5,0x21,0x31,0x4b]
// CHECK: sub      w30, w29, wzr, uxtw        // encoding: [0xbe,0x43,0x3f,0x4b]
// CHECK: sub      w19, w17, w1, uxtx         // encoding: [0x33,0x62,0x21,0x4b]
// CHECK: sub      w2, w5, w1, sxtb           // encoding: [0xa2,0x80,0x21,0x4b]
// CHECK: sub      w26, wsp, w19, sxth        // encoding: [0xfa,0xa3,0x33,0x4b]
// CHECK: sub      wsp, w2, w3, sxtw          // encoding: [0x5f,0xc0,0x23,0x4b]
// CHECK: sub      w2, w3, w5, sxtx           // encoding: [0x62,0xe0,0x25,0x4b]

        // Adds
        adds x2, x4, w5, uxtb #2
        adds x20, sp, w19, uxth #4
        adds x12, x1, w20, uxtw
        adds x20, x3, x13, uxtx #0
        adds xzr, x25, w20, sxtb #3
        adds x18, sp, w19, sxth
        adds xzr, x2, w3, sxtw
        adds x3, x5, x9, sxtx #2
// CHECK: adds     x2, x4, w5, uxtb #2        // encoding: [0x82,0x08,0x25,0xab]
// CHECK: adds     x20, sp, w19, uxth #4      // encoding: [0xf4,0x33,0x33,0xab]
// CHECK: adds     x12, x1, w20, uxtw         // encoding: [0x2c,0x40,0x34,0xab]
// CHECK: adds     x20, x3, x13, uxtx         // encoding: [0x74,0x60,0x2d,0xab]
// CHECK: {{adds xzr,|cmn}} x25, w20, sxtb #3     // encoding: [0x3f,0x8f,0x34,0xab]
// CHECK: adds     x18, sp, w19, sxth         // encoding: [0xf2,0xa3,0x33,0xab]
// CHECK: {{adds xzr,|cmn}} x2, w3, sxtw          // encoding: [0x5f,0xc0,0x23,0xab]
// CHECK: adds     x3, x5, x9, sxtx #2        // encoding: [0xa3,0xe8,0x29,0xab]

        adds w2, w5, w7, uxtb
        adds w21, w15, w17, uxth
        adds w30, w29, wzr, uxtw
        adds w19, w17, w1, uxtx  // Goodness knows what this means
        adds w2, w5, w1, sxtb #1
        adds w26, wsp, w19, sxth
        adds wzr, w2, w3, sxtw
        adds w2, w3, w5, sxtx
// CHECK: adds     w2, w5, w7, uxtb           // encoding: [0xa2,0x00,0x27,0x2b]
// CHECK: adds     w21, w15, w17, uxth        // encoding: [0xf5,0x21,0x31,0x2b]
// CHECK: adds     w30, w29, wzr, uxtw        // encoding: [0xbe,0x43,0x3f,0x2b]
// CHECK: adds     w19, w17, w1, uxtx         // encoding: [0x33,0x62,0x21,0x2b]
// CHECK: adds     w2, w5, w1, sxtb #1        // encoding: [0xa2,0x84,0x21,0x2b]
// CHECK: adds     w26, wsp, w19, sxth        // encoding: [0xfa,0xa3,0x33,0x2b]
// CHECK: cmn w2, w3, sxtw          // encoding: [0x5f,0xc0,0x23,0x2b]
// CHECK: adds     w2, w3, w5, sxtx           // encoding: [0x62,0xe0,0x25,0x2b]

        // subs
        subs x2, x4, w5, uxtb #2
        subs x20, sp, w19, uxth #4
        subs x12, x1, w20, uxtw
        subs x20, x3, x13, uxtx #0
        subs xzr, x25, w20, sxtb #3
        subs x18, sp, w19, sxth
        subs xzr, x2, w3, sxtw
        subs x3, x5, x9, sxtx #2
// CHECK: subs     x2, x4, w5, uxtb #2        // encoding: [0x82,0x08,0x25,0xeb]
// CHECK: subs     x20, sp, w19, uxth #4      // encoding: [0xf4,0x33,0x33,0xeb]
// CHECK: subs     x12, x1, w20, uxtw         // encoding: [0x2c,0x40,0x34,0xeb]
// CHECK: subs     x20, x3, x13, uxtx         // encoding: [0x74,0x60,0x2d,0xeb]
// CHECK: {{subs xzr,|cmp}} x25, w20, sxtb #3     // encoding: [0x3f,0x8f,0x34,0xeb]
// CHECK: subs     x18, sp, w19, sxth         // encoding: [0xf2,0xa3,0x33,0xeb]
// CHECK: {{subs xzr,|cmp}} x2, w3, sxtw          // encoding: [0x5f,0xc0,0x23,0xeb]
// CHECK: subs     x3, x5, x9, sxtx #2        // encoding: [0xa3,0xe8,0x29,0xeb]

        subs w2, w5, w7, uxtb
        subs w21, w15, w17, uxth
        subs w30, w29, wzr, uxtw
        subs w19, w17, w1, uxtx  // Goodness knows what this means
        subs w2, w5, w1, sxtb #1
        subs w26, wsp, w19, sxth
        subs wzr, w2, w3, sxtw
        subs w2, w3, w5, sxtx
// CHECK: subs     w2, w5, w7, uxtb           // encoding: [0xa2,0x00,0x27,0x6b]
// CHECK: subs     w21, w15, w17, uxth        // encoding: [0xf5,0x21,0x31,0x6b]
// CHECK: subs     w30, w29, wzr, uxtw        // encoding: [0xbe,0x43,0x3f,0x6b]
// CHECK: subs     w19, w17, w1, uxtx         // encoding: [0x33,0x62,0x21,0x6b]
// CHECK: subs     w2, w5, w1, sxtb #1        // encoding: [0xa2,0x84,0x21,0x6b]
// CHECK: subs     w26, wsp, w19, sxth        // encoding: [0xfa,0xa3,0x33,0x6b]
// CHECK: {{subs wzr,|cmp}} w2, w3, sxtw          // encoding: [0x5f,0xc0,0x23,0x6b]
// CHECK: subs     w2, w3, w5, sxtx           // encoding: [0x62,0xe0,0x25,0x6b]

        // cmp
        cmp x4, w5, uxtb #2
        cmp sp, w19, uxth #4
        cmp x1, w20, uxtw
        cmp x3, x13, uxtx #0
        cmp x25, w20, sxtb #3
        cmp sp, w19, sxth
        cmp x2, w3, sxtw
        cmp x5, x9, sxtx #2
// CHECK: cmp      x4, w5, uxtb #2            // encoding: [0x9f,0x08,0x25,0xeb]
// CHECK: cmp      sp, w19, uxth #4           // encoding: [0xff,0x33,0x33,0xeb]
// CHECK: cmp      x1, w20, uxtw              // encoding: [0x3f,0x40,0x34,0xeb]
// CHECK: cmp      x3, x13, uxtx              // encoding: [0x7f,0x60,0x2d,0xeb]
// CHECK: cmp      x25, w20, sxtb #3          // encoding: [0x3f,0x8f,0x34,0xeb]
// CHECK: cmp      sp, w19, sxth              // encoding: [0xff,0xa3,0x33,0xeb]
// CHECK: cmp      x2, w3, sxtw               // encoding: [0x5f,0xc0,0x23,0xeb]
// CHECK: cmp      x5, x9, sxtx #2            // encoding: [0xbf,0xe8,0x29,0xeb]

        cmp w5, w7, uxtb
        cmp w15, w17, uxth
        cmp w29, wzr, uxtw
        cmp w17, w1, uxtx  // Goodness knows what this means
        cmp w5, w1, sxtb #1
        cmp wsp, w19, sxth
        cmp w2, w3, sxtw
        cmp w3, w5, sxtx
// CHECK: cmp      w5, w7, uxtb               // encoding: [0xbf,0x00,0x27,0x6b]
// CHECK: cmp      w15, w17, uxth             // encoding: [0xff,0x21,0x31,0x6b]
// CHECK: cmp      w29, wzr, uxtw             // encoding: [0xbf,0x43,0x3f,0x6b]
// CHECK: cmp      w17, w1, uxtx              // encoding: [0x3f,0x62,0x21,0x6b]
// CHECK: cmp      w5, w1, sxtb #1            // encoding: [0xbf,0x84,0x21,0x6b]
// CHECK: cmp      wsp, w19, sxth             // encoding: [0xff,0xa3,0x33,0x6b]
// CHECK: cmp      w2, w3, sxtw               // encoding: [0x5f,0xc0,0x23,0x6b]
// CHECK: cmp      w3, w5, sxtx               // encoding: [0x7f,0xe0,0x25,0x6b]


        // cmn
        cmn x4, w5, uxtb #2
        cmn sp, w19, uxth #4
        cmn x1, w20, uxtw
        cmn x3, x13, uxtx #0
        cmn x25, w20, sxtb #3
        cmn sp, w19, sxth
        cmn x2, w3, sxtw
        cmn x5, x9, sxtx #2
// CHECK: cmn      x4, w5, uxtb #2            // encoding: [0x9f,0x08,0x25,0xab]
// CHECK: cmn      sp, w19, uxth #4           // encoding: [0xff,0x33,0x33,0xab]
// CHECK: cmn      x1, w20, uxtw              // encoding: [0x3f,0x40,0x34,0xab]
// CHECK: cmn      x3, x13, uxtx              // encoding: [0x7f,0x60,0x2d,0xab]
// CHECK: cmn      x25, w20, sxtb #3          // encoding: [0x3f,0x8f,0x34,0xab]
// CHECK: cmn      sp, w19, sxth              // encoding: [0xff,0xa3,0x33,0xab]
// CHECK: cmn      x2, w3, sxtw               // encoding: [0x5f,0xc0,0x23,0xab]
// CHECK: cmn      x5, x9, sxtx #2            // encoding: [0xbf,0xe8,0x29,0xab]

        cmn w5, w7, uxtb
        cmn w15, w17, uxth
        cmn w29, wzr, uxtw
        cmn w17, w1, uxtx  // Goodness knows what this means
        cmn w5, w1, sxtb #1
        cmn wsp, w19, sxth
        cmn w2, w3, sxtw
        cmn w3, w5, sxtx
// CHECK: {{cmn|adds wzr,}}      w5, w7, uxtb               // encoding: [0xbf,0x00,0x27,0x2b]
// CHECK: {{cmn|adds wzr,}}      w15, w17, uxth             // encoding: [0xff,0x21,0x31,0x2b]
// CHECK: {{cmn|adds wzr,}}      w29, wzr, uxtw             // encoding: [0xbf,0x43,0x3f,0x2b]
// CHECK: {{cmn|adds wzr,}}      w17, w1, uxtx              // encoding: [0x3f,0x62,0x21,0x2b]
// CHECK: {{cmn|adds wzr,}}      w5, w1, sxtb #1            // encoding: [0xbf,0x84,0x21,0x2b]
// CHECK: {{cmn|adds wzr,}}      wsp, w19, sxth             // encoding: [0xff,0xa3,0x33,0x2b]
// CHECK: {{cmn|adds wzr,}}      w2, w3, sxtw               // encoding: [0x5f,0xc0,0x23,0x2b]
// CHECK: {{cmn|adds wzr,}}      w3, w5, sxtx               // encoding: [0x7f,0xe0,0x25,0x2b]

        // operands for cmp
        cmp x20, w29, uxtb #3
        cmp x12, x13, uxtx #4
        cmp wsp, w1, uxtb
        cmn wsp, wzr, sxtw
// CHECK: cmp      x20, w29, uxtb #3          // encoding: [0x9f,0x0e,0x3d,0xeb]
// CHECK: cmp      x12, x13, uxtx #4          // encoding: [0x9f,0x71,0x2d,0xeb]
// CHECK: cmp      wsp, w1, uxtb              // encoding: [0xff,0x03,0x21,0x6b]
// CHECK: {{cmn|adds wzr,}}      wsp, wzr, sxtw             // encoding: [0xff,0xc3,0x3f,0x2b]

        // LSL variant if sp involved
        sub sp, x3, x7, lsl #4
        add w2, wsp, w3, lsl #1
        cmp wsp, w9, lsl #0
        adds wzr, wsp, w3, lsl #4
        subs x3, sp, x9, lsl #2
// CHECK: sub      sp, x3, x7, lsl #4         // encoding: [0x7f,0x70,0x27,0xcb]
// CHECK: add      w2, wsp, w3, lsl #1        // encoding: [0xe2,0x47,0x23,0x0b]
// CHECK: cmp      wsp, w9                    // encoding: [0xff,0x43,0x29,0x6b]
// CHECK: cmn wsp, w3, lsl #4       // encoding: [0xff,0x53,0x23,0x2b]
// CHECK: subs     x3, sp, x9, lsl #2         // encoding: [0xe3,0x6b,0x29,0xeb]

//------------------------------------------------------------------------------
// Add/sub (immediate)
//------------------------------------------------------------------------------

// Check basic immediate values: an unsigned 12-bit immediate, optionally
// shifted left by 12 bits.
        add w4, w5, #0x0
        add w2, w3, #4095
        add w30, w29, #1, lsl #12
        add w13, w5, #4095, lsl #12
        add x5, x7, #1638
// CHECK: add      w4, w5, #0                 // encoding: [0xa4,0x00,0x00,0x11]
// CHECK: add      w2, w3, #4095              // encoding: [0x62,0xfc,0x3f,0x11]
// CHECK: add      w30, w29, #1, lsl #12      // encoding: [0xbe,0x07,0x40,0x11]
// CHECK: add      w13, w5, #4095, lsl #12    // encoding: [0xad,0xfc,0x7f,0x11]
// CHECK: add      x5, x7, #1638              // encoding: [0xe5,0x98,0x19,0x91]

// All registers involved in the non-S variants have 31 encoding sp rather than zr
        add w20, wsp, #801, lsl #0
        add wsp, wsp, #1104
        add wsp, w30, #4084
// CHECK: add      w20, wsp, #801             // encoding: [0xf4,0x87,0x0c,0x11]
// CHECK: add      wsp, wsp, #1104            // encoding: [0xff,0x43,0x11,0x11]
// CHECK: add      wsp, w30, #4084            // encoding: [0xdf,0xd3,0x3f,0x11]

// A few checks on the sanity of 64-bit versions
        add x0, x24, #291
        add x3, x24, #4095, lsl #12
        add x8, sp, #1074
        add sp, x29, #3816
// CHECK: add      x0, x24, #291              // encoding: [0x00,0x8f,0x04,0x91]
// CHECK: add      x3, x24, #4095, lsl #12    // encoding: [0x03,0xff,0x7f,0x91]
// CHECK: add      x8, sp, #1074              // encoding: [0xe8,0xcb,0x10,0x91]
// CHECK: add      sp, x29, #3816             // encoding: [0xbf,0xa3,0x3b,0x91]

// And on sub
        sub w0, wsp, #4077
        sub w4, w20, #546, lsl #12
        sub sp, sp, #288
        sub wsp, w19, #16
// CHECK: sub      w0, wsp, #4077             // encoding: [0xe0,0xb7,0x3f,0x51]
// CHECK: sub      w4, w20, #546, lsl #12     // encoding: [0x84,0x8a,0x48,0x51]
// CHECK: sub      sp, sp, #288               // encoding: [0xff,0x83,0x04,0xd1]
// CHECK: sub      wsp, w19, #16              // encoding: [0x7f,0x42,0x00,0x51]

// ADDS/SUBS accept zr in the Rd position but sp in the Rn position
        adds w13, w23, #291, lsl #12
        adds wzr, w2, #4095                  // FIXME: canonically should be cmn
        adds w20, wsp, #0x0
        adds xzr, x3, #0x1, lsl #12          // FIXME: canonically should be cmn
// CHECK: adds     w13, w23, #291, lsl #12    // encoding: [0xed,0x8e,0x44,0x31]
// CHECK: {{adds wzr,|cmn}} w2, #4095         // encoding: [0x5f,0xfc,0x3f,0x31]
// CHECK: adds     w20, wsp, #0               // encoding: [0xf4,0x03,0x00,0x31]
// CHECK: {{adds xzr,|cmn}} x3, #1, lsl #12   // encoding: [0x7f,0x04,0x40,0xb1]

// Checks for subs
        subs xzr, sp, #20, lsl #12           // FIXME: canonically should be cmp
        subs xzr, x30, #4095, lsl #0         // FIXME: canonically should be cmp
        subs x4, sp, #3822
// CHECK: {{subs xzr,|cmp}} sp, #20, lsl #12  // encoding: [0xff,0x53,0x40,0xf1]
// CHECK: {{subs xzr,|cmp}} x30, #4095        // encoding: [0xdf,0xff,0x3f,0xf1]
// CHECK: subs     x4, sp, #3822              // encoding: [0xe4,0xbb,0x3b,0xf1]

// cmn is an alias for adds zr, ...
        cmn w3, #291, lsl #12
        cmn wsp, #1365, lsl #0
        cmn sp, #1092, lsl #12
// CHECK: cmn      w3, #291, lsl #12          // encoding: [0x7f,0x8c,0x44,0x31]
// CHECK: cmn      wsp, #1365                 // encoding: [0xff,0x57,0x15,0x31]
// CHECK: cmn      sp, #1092, lsl #12         // encoding: [0xff,0x13,0x51,0xb1]

// cmp is an alias for subs zr, ... (FIXME: should always disassemble as such too).
        cmp x4, #300, lsl #12
        cmp wsp, #500
        cmp sp, #200, lsl #0
// CHECK: cmp      x4, #300, lsl #12          // encoding: [0x9f,0xb0,0x44,0xf1]
// CHECK: cmp      wsp, #500                  // encoding: [0xff,0xd3,0x07,0x71]
// CHECK: cmp      sp, #200                   // encoding: [0xff,0x23,0x03,0xf1]

// A "MOV" involving sp is encoded in this manner: add Reg, Reg, #0
        mov sp, x30
        mov wsp, w20
        mov x11, sp
        mov w24, wsp
// CHECK: mov      sp, x30                    // encoding: [0xdf,0x03,0x00,0x91]
// CHECK: mov      wsp, w20                   // encoding: [0x9f,0x02,0x00,0x11]
// CHECK: mov      x11, sp                    // encoding: [0xeb,0x03,0x00,0x91]
// CHECK: mov      w24, wsp                   // encoding: [0xf8,0x03,0x00,0x11]

// A relocation check (default to lo12, which is the only sane relocation anyway really)
        add x0, x4, #:lo12:var
// CHECK: add x0, x4, :lo12:var       // encoding: [0x80,0bAAAAAA00,0b00AAAAAA,0x91]
// CHECK:                             // fixup A - offset: 0, value: :lo12:var, kind: fixup_aarch64_add_imm12

//------------------------------------------------------------------------------
// Add-sub (shifted register)
//------------------------------------------------------------------------------

// As usual, we don't print the canonical forms of many instructions.

        add w3, w5, w7
        add wzr, w3, w5
        add w20, wzr, w4
        add w4, w6, wzr
// CHECK: add      w3, w5, w7                 // encoding: [0xa3,0x00,0x07,0x0b]
// CHECK: add      wzr, w3, w5                // encoding: [0x7f,0x00,0x05,0x0b]
// CHECK: add      w20, wzr, w4               // encoding: [0xf4,0x03,0x04,0x0b]
// CHECK: add      w4, w6, wzr                // encoding: [0xc4,0x00,0x1f,0x0b]

        add w11, w13, w15, lsl #0
        add w9, w3, wzr, lsl #10
        add w17, w29, w20, lsl #31
        add w17, w29, w20, lsl #(31-2)
// CHECK: add      w11, w13, w15              // encoding: [0xab,0x01,0x0f,0x0b]
// CHECK: add      w9, w3, wzr, lsl #10       // encoding: [0x69,0x28,0x1f,0x0b]
// CHECK: add      w17, w29, w20, lsl #31     // encoding: [0xb1,0x7f,0x14,0x0b]
// CHECK: add      w17, w29, w20, lsl #29     // encoding: [0xb1,0x77,0x14,0x0b]

        add w21, w22, w23, lsr #0
        add w24, w25, w26, lsr #18
        add w27, w28, w29, lsr #31
        add w27, w28, w29, lsr #(31-2)
// CHECK: add      w21, w22, w23, lsr #0      // encoding: [0xd5,0x02,0x57,0x0b]
// CHECK: add      w24, w25, w26, lsr #18     // encoding: [0x38,0x4b,0x5a,0x0b]
// CHECK: add      w27, w28, w29, lsr #31     // encoding: [0x9b,0x7f,0x5d,0x0b]
// CHECK: add      w27, w28, w29, lsr #29     // encoding: [0x9b,0x77,0x5d,0x0b]

        add w2, w3, w4, asr #0
        add w5, w6, w7, asr #21
        add w8, w9, w10, asr #31
        add w8, w9, w10, asr #(31-2)
// CHECK: add      w2, w3, w4, asr #0         // encoding: [0x62,0x00,0x84,0x0b]
// CHECK: add      w5, w6, w7, asr #21        // encoding: [0xc5,0x54,0x87,0x0b]
// CHECK: add      w8, w9, w10, asr #31       // encoding: [0x28,0x7d,0x8a,0x0b]
// CHECK: add      w8, w9, w10, asr #29       // encoding: [0x28,0x75,0x8a,0x0b]

        add x3, x5, x7
        add xzr, x3, x5
        add x20, xzr, x4
        add x4, x6, xzr
// CHECK: add      x3, x5, x7                 // encoding: [0xa3,0x00,0x07,0x8b]
// CHECK: add      xzr, x3, x5                // encoding: [0x7f,0x00,0x05,0x8b]
// CHECK: add      x20, xzr, x4               // encoding: [0xf4,0x03,0x04,0x8b]
// CHECK: add      x4, x6, xzr                // encoding: [0xc4,0x00,0x1f,0x8b]

        add x11, x13, x15, lsl #0
        add x9, x3, xzr, lsl #10
        add x17, x29, x20, lsl #63
        add x17, x29, x20, lsl #(63-5)
// CHECK: add      x11, x13, x15              // encoding: [0xab,0x01,0x0f,0x8b]
// CHECK: add      x9, x3, xzr, lsl #10       // encoding: [0x69,0x28,0x1f,0x8b]
// CHECK: add      x17, x29, x20, lsl #63     // encoding: [0xb1,0xff,0x14,0x8b]
// CHECK: add	   x17, x29, x20, lsl #58     // encoding: [0xb1,0xeb,0x14,0x8b]

        add x21, x22, x23, lsr #0
        add x24, x25, x26, lsr #18
        add x27, x28, x29, lsr #63
        add x17, x29, x20, lsr #(63-5)
// CHECK: add      x21, x22, x23, lsr #0      // encoding: [0xd5,0x02,0x57,0x8b]
// CHECK: add      x24, x25, x26, lsr #18     // encoding: [0x38,0x4b,0x5a,0x8b]
// CHECK: add      x27, x28, x29, lsr #63     // encoding: [0x9b,0xff,0x5d,0x8b]
// CHECK: add	   x17, x29, x20, lsr #58     // encoding: [0xb1,0xeb,0x54,0x8b]

        add x2, x3, x4, asr #0
        add x5, x6, x7, asr #21
        add x8, x9, x10, asr #63
        add x17, x29, x20, asr #(63-5)
// CHECK: add      x2, x3, x4, asr #0         // encoding: [0x62,0x00,0x84,0x8b]
// CHECK: add      x5, x6, x7, asr #21        // encoding: [0xc5,0x54,0x87,0x8b]
// CHECK: add      x8, x9, x10, asr #63       // encoding: [0x28,0xfd,0x8a,0x8b]
// CHECK: add	   x17, x29, x20, asr #58     // encoding: [0xb1,0xeb,0x94,0x8b]

        adds w3, w5, w7
        adds wzr, w3, w5
        adds w20, wzr, w4
        adds w4, w6, wzr
// CHECK: adds     w3, w5, w7                 // encoding: [0xa3,0x00,0x07,0x2b]
// CHECK: {{adds wzr,|cmn}} w3, w5                // encoding: [0x7f,0x00,0x05,0x2b]
// CHECK: adds     w20, wzr, w4               // encoding: [0xf4,0x03,0x04,0x2b]
// CHECK: adds     w4, w6, wzr                // encoding: [0xc4,0x00,0x1f,0x2b]

        adds w11, w13, w15, lsl #0
        adds w9, w3, wzr, lsl #10
        adds w17, w29, w20, lsl #31
// CHECK: adds     w11, w13, w15              // encoding: [0xab,0x01,0x0f,0x2b]
// CHECK: adds     w9, w3, wzr, lsl #10       // encoding: [0x69,0x28,0x1f,0x2b]
// CHECK: adds     w17, w29, w20, lsl #31     // encoding: [0xb1,0x7f,0x14,0x2b]

        adds w21, w22, w23, lsr #0
        adds w24, w25, w26, lsr #18
        adds w27, w28, w29, lsr #31
// CHECK: adds     w21, w22, w23, lsr #0      // encoding: [0xd5,0x02,0x57,0x2b]
// CHECK: adds     w24, w25, w26, lsr #18     // encoding: [0x38,0x4b,0x5a,0x2b]
// CHECK: adds     w27, w28, w29, lsr #31     // encoding: [0x9b,0x7f,0x5d,0x2b]

        adds w2, w3, w4, asr #0
        adds w5, w6, w7, asr #21
        adds w8, w9, w10, asr #31
// CHECK: adds     w2, w3, w4, asr #0         // encoding: [0x62,0x00,0x84,0x2b]
// CHECK: adds     w5, w6, w7, asr #21        // encoding: [0xc5,0x54,0x87,0x2b]
// CHECK: adds     w8, w9, w10, asr #31       // encoding: [0x28,0x7d,0x8a,0x2b]

        adds x3, x5, x7
        adds xzr, x3, x5
        adds x20, xzr, x4
        adds x4, x6, xzr
// CHECK: adds     x3, x5, x7                 // encoding: [0xa3,0x00,0x07,0xab]
// CHECK: {{adds xzr,|cmn}} x3, x5                // encoding: [0x7f,0x00,0x05,0xab]
// CHECK: adds     x20, xzr, x4               // encoding: [0xf4,0x03,0x04,0xab]
// CHECK: adds     x4, x6, xzr                // encoding: [0xc4,0x00,0x1f,0xab]

        adds x11, x13, x15, lsl #0
        adds x9, x3, xzr, lsl #10
        adds x17, x29, x20, lsl #63
// CHECK: adds     x11, x13, x15              // encoding: [0xab,0x01,0x0f,0xab]
// CHECK: adds     x9, x3, xzr, lsl #10       // encoding: [0x69,0x28,0x1f,0xab]
// CHECK: adds     x17, x29, x20, lsl #63     // encoding: [0xb1,0xff,0x14,0xab]

        adds x21, x22, x23, lsr #0
        adds x24, x25, x26, lsr #18
        adds x27, x28, x29, lsr #63
// CHECK: adds     x21, x22, x23, lsr #0      // encoding: [0xd5,0x02,0x57,0xab]
// CHECK: adds     x24, x25, x26, lsr #18     // encoding: [0x38,0x4b,0x5a,0xab]
// CHECK: adds     x27, x28, x29, lsr #63     // encoding: [0x9b,0xff,0x5d,0xab]

        adds x2, x3, x4, asr #0
        adds x5, x6, x7, asr #21
        adds x8, x9, x10, asr #63
// CHECK: adds     x2, x3, x4, asr #0         // encoding: [0x62,0x00,0x84,0xab]
// CHECK: adds     x5, x6, x7, asr #21        // encoding: [0xc5,0x54,0x87,0xab]
// CHECK: adds     x8, x9, x10, asr #63       // encoding: [0x28,0xfd,0x8a,0xab]

        sub w3, w5, w7
        sub wzr, w3, w5
        sub w20, wzr, w4
        sub w4, w6, wzr
// CHECK: sub      w3, w5, w7                 // encoding: [0xa3,0x00,0x07,0x4b]
// CHECK: sub      wzr, w3, w5                // encoding: [0x7f,0x00,0x05,0x4b]
// CHECK: neg      w20, w4              // encoding: [0xf4,0x03,0x04,0x4b]
// CHECK: sub      w4, w6, wzr                // encoding: [0xc4,0x00,0x1f,0x4b]

        sub w11, w13, w15, lsl #0
        sub w9, w3, wzr, lsl #10
        sub w17, w29, w20, lsl #31
// CHECK: sub      w11, w13, w15              // encoding: [0xab,0x01,0x0f,0x4b]
// CHECK: sub      w9, w3, wzr, lsl #10       // encoding: [0x69,0x28,0x1f,0x4b]
// CHECK: sub      w17, w29, w20, lsl #31     // encoding: [0xb1,0x7f,0x14,0x4b]

        sub w21, w22, w23, lsr #0
        sub w24, w25, w26, lsr #18
        sub w27, w28, w29, lsr #31
// CHECK: sub      w21, w22, w23, lsr #0      // encoding: [0xd5,0x02,0x57,0x4b]
// CHECK: sub      w24, w25, w26, lsr #18     // encoding: [0x38,0x4b,0x5a,0x4b]
// CHECK: sub      w27, w28, w29, lsr #31     // encoding: [0x9b,0x7f,0x5d,0x4b]

        sub w2, w3, w4, asr #0
        sub w5, w6, w7, asr #21
        sub w8, w9, w10, asr #31
// CHECK: sub      w2, w3, w4, asr #0         // encoding: [0x62,0x00,0x84,0x4b]
// CHECK: sub      w5, w6, w7, asr #21        // encoding: [0xc5,0x54,0x87,0x4b]
// CHECK: sub      w8, w9, w10, asr #31       // encoding: [0x28,0x7d,0x8a,0x4b]

        sub x3, x5, x7
        sub xzr, x3, x5
        sub x20, xzr, x4
        sub x4, x6, xzr
// CHECK: sub      x3, x5, x7                 // encoding: [0xa3,0x00,0x07,0xcb]
// CHECK: sub      xzr, x3, x5                // encoding: [0x7f,0x00,0x05,0xcb]
// CHECK: neg      x20, x4              // encoding: [0xf4,0x03,0x04,0xcb]
// CHECK: sub      x4, x6, xzr                // encoding: [0xc4,0x00,0x1f,0xcb]

        sub x11, x13, x15, lsl #0
        sub x9, x3, xzr, lsl #10
        sub x17, x29, x20, lsl #63
// CHECK: sub      x11, x13, x15              // encoding: [0xab,0x01,0x0f,0xcb]
// CHECK: sub      x9, x3, xzr, lsl #10       // encoding: [0x69,0x28,0x1f,0xcb]
// CHECK: sub      x17, x29, x20, lsl #63     // encoding: [0xb1,0xff,0x14,0xcb]

        sub x21, x22, x23, lsr #0
        sub x24, x25, x26, lsr #18
        sub x27, x28, x29, lsr #63
// CHECK: sub      x21, x22, x23, lsr #0      // encoding: [0xd5,0x02,0x57,0xcb]
// CHECK: sub      x24, x25, x26, lsr #18     // encoding: [0x38,0x4b,0x5a,0xcb]
// CHECK: sub      x27, x28, x29, lsr #63     // encoding: [0x9b,0xff,0x5d,0xcb]

        sub x2, x3, x4, asr #0
        sub x5, x6, x7, asr #21
        sub x8, x9, x10, asr #63
// CHECK: sub      x2, x3, x4, asr #0         // encoding: [0x62,0x00,0x84,0xcb]
// CHECK: sub      x5, x6, x7, asr #21        // encoding: [0xc5,0x54,0x87,0xcb]
// CHECK: sub      x8, x9, x10, asr #63       // encoding: [0x28,0xfd,0x8a,0xcb]

        subs w3, w5, w7
        subs wzr, w3, w5
        subs w20, wzr, w4
        subs w4, w6, wzr
// CHECK: subs     w3, w5, w7                 // encoding: [0xa3,0x00,0x07,0x6b]
// CHECK: {{subs wzr,|cmp}} w3, w5            // encoding: [0x7f,0x00,0x05,0x6b]
// CHECK: negs     w20, w4              // encoding: [0xf4,0x03,0x04,0x6b]
// CHECK: subs     w4, w6, wzr                // encoding: [0xc4,0x00,0x1f,0x6b]

        subs w11, w13, w15, lsl #0
        subs w9, w3, wzr, lsl #10
        subs w17, w29, w20, lsl #31
// CHECK: subs     w11, w13, w15              // encoding: [0xab,0x01,0x0f,0x6b]
// CHECK: subs     w9, w3, wzr, lsl #10       // encoding: [0x69,0x28,0x1f,0x6b]
// CHECK: subs     w17, w29, w20, lsl #31     // encoding: [0xb1,0x7f,0x14,0x6b]

        subs w21, w22, w23, lsr #0
        subs w24, w25, w26, lsr #18
        subs w27, w28, w29, lsr #31
// CHECK: subs     w21, w22, w23, lsr #0      // encoding: [0xd5,0x02,0x57,0x6b]
// CHECK: subs     w24, w25, w26, lsr #18     // encoding: [0x38,0x4b,0x5a,0x6b]
// CHECK: subs     w27, w28, w29, lsr #31     // encoding: [0x9b,0x7f,0x5d,0x6b]

        subs w2, w3, w4, asr #0
        subs w5, w6, w7, asr #21
        subs w8, w9, w10, asr #31
// CHECK: subs     w2, w3, w4, asr #0         // encoding: [0x62,0x00,0x84,0x6b]
// CHECK: subs     w5, w6, w7, asr #21        // encoding: [0xc5,0x54,0x87,0x6b]
// CHECK: subs     w8, w9, w10, asr #31       // encoding: [0x28,0x7d,0x8a,0x6b]

        subs x3, x5, x7
        subs xzr, x3, x5
        subs x20, xzr, x4
        subs x4, x6, xzr
// CHECK: subs     x3, x5, x7                 // encoding: [0xa3,0x00,0x07,0xeb]
// CHECK: {{subs xzr,|cmp}} x3, x5            // encoding: [0x7f,0x00,0x05,0xeb]
// CHECK: negs     x20, x4              // encoding: [0xf4,0x03,0x04,0xeb]
// CHECK: subs     x4, x6, xzr                // encoding: [0xc4,0x00,0x1f,0xeb]

        subs x11, x13, x15, lsl #0
        subs x9, x3, xzr, lsl #10
        subs x17, x29, x20, lsl #63
// CHECK: subs     x11, x13, x15              // encoding: [0xab,0x01,0x0f,0xeb]
// CHECK: subs     x9, x3, xzr, lsl #10       // encoding: [0x69,0x28,0x1f,0xeb]
// CHECK: subs     x17, x29, x20, lsl #63     // encoding: [0xb1,0xff,0x14,0xeb]

        subs x21, x22, x23, lsr #0
        subs x24, x25, x26, lsr #18
        subs x27, x28, x29, lsr #63
// CHECK: subs     x21, x22, x23, lsr #0      // encoding: [0xd5,0x02,0x57,0xeb]
// CHECK: subs     x24, x25, x26, lsr #18     // encoding: [0x38,0x4b,0x5a,0xeb]
// CHECK: subs     x27, x28, x29, lsr #63     // encoding: [0x9b,0xff,0x5d,0xeb]

        subs x2, x3, x4, asr #0
        subs x5, x6, x7, asr #21
        subs x8, x9, x10, asr #63
// CHECK: subs     x2, x3, x4, asr #0         // encoding: [0x62,0x00,0x84,0xeb]
// CHECK: subs     x5, x6, x7, asr #21        // encoding: [0xc5,0x54,0x87,0xeb]
// CHECK: subs     x8, x9, x10, asr #63       // encoding: [0x28,0xfd,0x8a,0xeb]

        cmn w0, w3
        cmn wzr, w4
        cmn w5, wzr
        cmn wsp, w6
// CHECK: cmn      w0, w3                     // encoding: [0x1f,0x00,0x03,0x2b]
// CHECK: cmn      wzr, w4                    // encoding: [0xff,0x03,0x04,0x2b]
// CHECK: cmn      w5, wzr                    // encoding: [0xbf,0x00,0x1f,0x2b]
// CHECK: cmn      wsp, w6                    // encoding: [0xff,0x43,0x26,0x2b]

        cmn w6, w7, lsl #0
        cmn w8, w9, lsl #15
        cmn w10, w11, lsl #31
// CHECK: cmn      w6, w7                     // encoding: [0xdf,0x00,0x07,0x2b]
// CHECK: cmn      w8, w9, lsl #15            // encoding: [0x1f,0x3d,0x09,0x2b]
// CHECK: cmn      w10, w11, lsl #31          // encoding: [0x5f,0x7d,0x0b,0x2b]

        cmn w12, w13, lsr #0
        cmn w14, w15, lsr #21
        cmn w16, w17, lsr #31
// CHECK: cmn      w12, w13, lsr #0           // encoding: [0x9f,0x01,0x4d,0x2b]
// CHECK: cmn      w14, w15, lsr #21          // encoding: [0xdf,0x55,0x4f,0x2b]
// CHECK: cmn      w16, w17, lsr #31          // encoding: [0x1f,0x7e,0x51,0x2b]

        cmn w18, w19, asr #0
        cmn w20, w21, asr #22
        cmn w22, w23, asr #31
// CHECK: cmn      w18, w19, asr #0           // encoding: [0x5f,0x02,0x93,0x2b]
// CHECK: cmn      w20, w21, asr #22          // encoding: [0x9f,0x5a,0x95,0x2b]
// CHECK: cmn      w22, w23, asr #31          // encoding: [0xdf,0x7e,0x97,0x2b]

        cmn x0, x3
        cmn xzr, x4
        cmn x5, xzr
        cmn sp, x6
// CHECK: cmn      x0, x3                     // encoding: [0x1f,0x00,0x03,0xab]
// CHECK: cmn      xzr, x4                    // encoding: [0xff,0x03,0x04,0xab]
// CHECK: cmn      x5, xzr                    // encoding: [0xbf,0x00,0x1f,0xab]
// CHECK: cmn      sp, x6                     // encoding: [0xff,0x63,0x26,0xab]

        cmn x6, x7, lsl #0
        cmn x8, x9, lsl #15
        cmn x10, x11, lsl #63
// CHECK: cmn      x6, x7                     // encoding: [0xdf,0x00,0x07,0xab]
// CHECK: cmn      x8, x9, lsl #15            // encoding: [0x1f,0x3d,0x09,0xab]
// CHECK: cmn      x10, x11, lsl #63          // encoding: [0x5f,0xfd,0x0b,0xab]

        cmn x12, x13, lsr #0
        cmn x14, x15, lsr #41
        cmn x16, x17, lsr #63
// CHECK: cmn      x12, x13, lsr #0           // encoding: [0x9f,0x01,0x4d,0xab]
// CHECK: cmn      x14, x15, lsr #41          // encoding: [0xdf,0xa5,0x4f,0xab]
// CHECK: cmn      x16, x17, lsr #63          // encoding: [0x1f,0xfe,0x51,0xab]

        cmn x18, x19, asr #0
        cmn x20, x21, asr #55
        cmn x22, x23, asr #63
// CHECK: cmn      x18, x19, asr #0           // encoding: [0x5f,0x02,0x93,0xab]
// CHECK: cmn      x20, x21, asr #55          // encoding: [0x9f,0xde,0x95,0xab]
// CHECK: cmn      x22, x23, asr #63          // encoding: [0xdf,0xfe,0x97,0xab]

        cmp w0, w3
        cmp wzr, w4
        cmp w5, wzr
        cmp wsp, w6
// CHECK: cmp      w0, w3                     // encoding: [0x1f,0x00,0x03,0x6b]
// CHECK: cmp      wzr, w4                    // encoding: [0xff,0x03,0x04,0x6b]
// CHECK: cmp      w5, wzr                    // encoding: [0xbf,0x00,0x1f,0x6b]
// CHECK: cmp      wsp, w6                    // encoding: [0xff,0x43,0x26,0x6b]

        cmp w6, w7, lsl #0
        cmp w8, w9, lsl #15
        cmp w10, w11, lsl #31
// CHECK: cmp      w6, w7                     // encoding: [0xdf,0x00,0x07,0x6b]
// CHECK: cmp      w8, w9, lsl #15            // encoding: [0x1f,0x3d,0x09,0x6b]
// CHECK: cmp      w10, w11, lsl #31          // encoding: [0x5f,0x7d,0x0b,0x6b]

        cmp w12, w13, lsr #0
        cmp w14, w15, lsr #21
        cmp w16, w17, lsr #31
// CHECK: cmp      w12, w13, lsr #0           // encoding: [0x9f,0x01,0x4d,0x6b]
// CHECK: cmp      w14, w15, lsr #21          // encoding: [0xdf,0x55,0x4f,0x6b]
// CHECK: cmp      w16, w17, lsr #31          // encoding: [0x1f,0x7e,0x51,0x6b]

        cmp w18, w19, asr #0
        cmp w20, w21, asr #22
        cmp w22, w23, asr #31
// CHECK: cmp      w18, w19, asr #0           // encoding: [0x5f,0x02,0x93,0x6b]
// CHECK: cmp      w20, w21, asr #22          // encoding: [0x9f,0x5a,0x95,0x6b]
// CHECK: cmp      w22, w23, asr #31          // encoding: [0xdf,0x7e,0x97,0x6b]

        cmp x0, x3
        cmp xzr, x4
        cmp x5, xzr
        cmp sp, x6
// CHECK: cmp      x0, x3                     // encoding: [0x1f,0x00,0x03,0xeb]
// CHECK: cmp      xzr, x4                    // encoding: [0xff,0x03,0x04,0xeb]
// CHECK: cmp      x5, xzr                    // encoding: [0xbf,0x00,0x1f,0xeb]
// CHECK: cmp      sp, x6                     // encoding: [0xff,0x63,0x26,0xeb]

        cmp x6, x7, lsl #0
        cmp x8, x9, lsl #15
        cmp x10, x11, lsl #63
// CHECK: cmp      x6, x7                     // encoding: [0xdf,0x00,0x07,0xeb]
// CHECK: cmp      x8, x9, lsl #15            // encoding: [0x1f,0x3d,0x09,0xeb]
// CHECK: cmp      x10, x11, lsl #63          // encoding: [0x5f,0xfd,0x0b,0xeb]

        cmp x12, x13, lsr #0
        cmp x14, x15, lsr #41
        cmp x16, x17, lsr #63
// CHECK: cmp      x12, x13, lsr #0           // encoding: [0x9f,0x01,0x4d,0xeb]
// CHECK: cmp      x14, x15, lsr #41          // encoding: [0xdf,0xa5,0x4f,0xeb]
// CHECK: cmp      x16, x17, lsr #63          // encoding: [0x1f,0xfe,0x51,0xeb]

        cmp x18, x19, asr #0
        cmp x20, x21, asr #55
        cmp x22, x23, asr #63
// CHECK: cmp      x18, x19, asr #0           // encoding: [0x5f,0x02,0x93,0xeb]
// CHECK: cmp      x20, x21, asr #55          // encoding: [0x9f,0xde,0x95,0xeb]
// CHECK: cmp      x22, x23, asr #63          // encoding: [0xdf,0xfe,0x97,0xeb]

        neg w29, w30
        neg w30, wzr
        neg wzr, w0
// CHECK: neg      w29, w30              // encoding: [0xfd,0x03,0x1e,0x4b]
// CHECK: neg      w30, wzr              // encoding: [0xfe,0x03,0x1f,0x4b]
// CHECK: neg      wzr, w0               // encoding: [0xff,0x03,0x00,0x4b]

        neg w28, w27, lsl #0
        neg w26, w25, lsl #29
        neg w24, w23, lsl #31

// CHECK: neg      w28, w27              // encoding: [0xfc,0x03,0x1b,0x4b]
// CHECK: neg      w26, w25, lsl #29     // encoding: [0xfa,0x77,0x19,0x4b]
// CHECK: neg      w24, w23, lsl #31     // encoding: [0xf8,0x7f,0x17,0x4b]

        neg w22, w21, lsr #0
        neg w20, w19, lsr #1
        neg w18, w17, lsr #31
// CHECK: neg      w22, w21, lsr #0      // encoding: [0xf6,0x03,0x55,0x4b]
// CHECK: neg      w20, w19, lsr #1      // encoding: [0xf4,0x07,0x53,0x4b]
// CHECK: neg      w18, w17, lsr #31     // encoding: [0xf2,0x7f,0x51,0x4b]

        neg w16, w15, asr #0
        neg w14, w13, asr #12
        neg w12, w11, asr #31
// CHECK: neg      w16, w15, asr #0      // encoding: [0xf0,0x03,0x8f,0x4b]
// CHECK: neg      w14, w13, asr #12     // encoding: [0xee,0x33,0x8d,0x4b]
// CHECK: neg      w12, w11, asr #31     // encoding: [0xec,0x7f,0x8b,0x4b]

        neg x29, x30
        neg x30, xzr
        neg xzr, x0
// CHECK: neg      x29, x30              // encoding: [0xfd,0x03,0x1e,0xcb]
// CHECK: neg      x30, xzr              // encoding: [0xfe,0x03,0x1f,0xcb]
// CHECK: neg      xzr, x0               // encoding: [0xff,0x03,0x00,0xcb]

        neg x28, x27, lsl #0
        neg x26, x25, lsl #29
        neg x24, x23, lsl #31

// CHECK: neg      x28, x27              // encoding: [0xfc,0x03,0x1b,0xcb]
// CHECK: neg      x26, x25, lsl #29     // encoding: [0xfa,0x77,0x19,0xcb]
// CHECK: neg      x24, x23, lsl #31     // encoding: [0xf8,0x7f,0x17,0xcb]

        neg x22, x21, lsr #0
        neg x20, x19, lsr #1
        neg x18, x17, lsr #31
// CHECK: neg      x22, x21, lsr #0      // encoding: [0xf6,0x03,0x55,0xcb]
// CHECK: neg      x20, x19, lsr #1      // encoding: [0xf4,0x07,0x53,0xcb]
// CHECK: neg      x18, x17, lsr #31     // encoding: [0xf2,0x7f,0x51,0xcb]

        neg x16, x15, asr #0
        neg x14, x13, asr #12
        neg x12, x11, asr #31
// CHECK: neg      x16, x15, asr #0      // encoding: [0xf0,0x03,0x8f,0xcb]
// CHECK: neg      x14, x13, asr #12     // encoding: [0xee,0x33,0x8d,0xcb]
// CHECK: neg      x12, x11, asr #31     // encoding: [0xec,0x7f,0x8b,0xcb]

        negs w29, w30
        negs w30, wzr
        negs wzr, w0
// CHECK: negs     w29, w30              // encoding: [0xfd,0x03,0x1e,0x6b]
// CHECK: negs     w30, wzr              // encoding: [0xfe,0x03,0x1f,0x6b]
// CHECK: cmp      wzr, w0               // encoding: [0xff,0x03,0x00,0x6b]

        negs w28, w27, lsl #0
        negs w26, w25, lsl #29
        negs w24, w23, lsl #31

// CHECK: negs     w28, w27             // encoding: [0xfc,0x03,0x1b,0x6b]
// CHECK: negs     w26, w25, lsl #29     // encoding: [0xfa,0x77,0x19,0x6b]
// CHECK: negs     w24, w23, lsl #31     // encoding: [0xf8,0x7f,0x17,0x6b]

        negs w22, w21, lsr #0
        negs w20, w19, lsr #1
        negs w18, w17, lsr #31
// CHECK: negs     w22, w21, lsr #0      // encoding: [0xf6,0x03,0x55,0x6b]
// CHECK: negs     w20, w19, lsr #1      // encoding: [0xf4,0x07,0x53,0x6b]
// CHECK: negs     w18, w17, lsr #31     // encoding: [0xf2,0x7f,0x51,0x6b]

        negs w16, w15, asr #0
        negs w14, w13, asr #12
        negs w12, w11, asr #31
// CHECK: negs     w16, w15, asr #0      // encoding: [0xf0,0x03,0x8f,0x6b]
// CHECK: negs     w14, w13, asr #12     // encoding: [0xee,0x33,0x8d,0x6b]
// CHECK: negs     w12, w11, asr #31     // encoding: [0xec,0x7f,0x8b,0x6b]

        negs x29, x30
        negs x30, xzr
        negs xzr, x0
// CHECK: negs     x29, x30              // encoding: [0xfd,0x03,0x1e,0xeb]
// CHECK: negs     x30, xzr              // encoding: [0xfe,0x03,0x1f,0xeb]
// CHECK: cmp     xzr, x0                // encoding: [0xff,0x03,0x00,0xeb]

        negs x28, x27, lsl #0
        negs x26, x25, lsl #29
        negs x24, x23, lsl #31

// CHECK: negs     x28, x27              // encoding: [0xfc,0x03,0x1b,0xeb]
// CHECK: negs     x26, x25, lsl #29     // encoding: [0xfa,0x77,0x19,0xeb]
// CHECK: negs     x24, x23, lsl #31     // encoding: [0xf8,0x7f,0x17,0xeb]

        negs x22, x21, lsr #0
        negs x20, x19, lsr #1
        negs x18, x17, lsr #31
// CHECK: negs     x22, x21, lsr #0      // encoding: [0xf6,0x03,0x55,0xeb]
// CHECK: negs     x20, x19, lsr #1      // encoding: [0xf4,0x07,0x53,0xeb]
// CHECK: negs     x18, x17, lsr #31     // encoding: [0xf2,0x7f,0x51,0xeb]

        negs x16, x15, asr #0
        negs x14, x13, asr #12
        negs x12, x11, asr #31
// CHECK: negs     x16, x15, asr #0      // encoding: [0xf0,0x03,0x8f,0xeb]
// CHECK: negs     x14, x13, asr #12     // encoding: [0xee,0x33,0x8d,0xeb]
// CHECK: negs     x12, x11, asr #31     // encoding: [0xec,0x7f,0x8b,0xeb]

//------------------------------------------------------------------------------
// Add-sub (shifted register)
//------------------------------------------------------------------------------
        adc w29, w27, w25
        adc wzr, w3, w4
        adc w9, wzr, w10
        adc w20, w0, wzr
// CHECK: adc      w29, w27, w25              // encoding: [0x7d,0x03,0x19,0x1a]
// CHECK: adc      wzr, w3, w4                // encoding: [0x7f,0x00,0x04,0x1a]
// CHECK: adc      w9, wzr, w10               // encoding: [0xe9,0x03,0x0a,0x1a]
// CHECK: adc      w20, w0, wzr               // encoding: [0x14,0x00,0x1f,0x1a]

        adc x29, x27, x25
        adc xzr, x3, x4
        adc x9, xzr, x10
        adc x20, x0, xzr
// CHECK: adc      x29, x27, x25              // encoding: [0x7d,0x03,0x19,0x9a]
// CHECK: adc      xzr, x3, x4                // encoding: [0x7f,0x00,0x04,0x9a]
// CHECK: adc      x9, xzr, x10               // encoding: [0xe9,0x03,0x0a,0x9a]
// CHECK: adc      x20, x0, xzr               // encoding: [0x14,0x00,0x1f,0x9a]

        adcs w29, w27, w25
        adcs wzr, w3, w4
        adcs w9, wzr, w10
        adcs w20, w0, wzr
// CHECK: adcs     w29, w27, w25              // encoding: [0x7d,0x03,0x19,0x3a]
// CHECK: adcs     wzr, w3, w4                // encoding: [0x7f,0x00,0x04,0x3a]
// CHECK: adcs     w9, wzr, w10               // encoding: [0xe9,0x03,0x0a,0x3a]
// CHECK: adcs     w20, w0, wzr               // encoding: [0x14,0x00,0x1f,0x3a]

        adcs x29, x27, x25
        adcs xzr, x3, x4
        adcs x9, xzr, x10
        adcs x20, x0, xzr
// CHECK: adcs     x29, x27, x25              // encoding: [0x7d,0x03,0x19,0xba]
// CHECK: adcs     xzr, x3, x4                // encoding: [0x7f,0x00,0x04,0xba]
// CHECK: adcs     x9, xzr, x10               // encoding: [0xe9,0x03,0x0a,0xba]
// CHECK: adcs     x20, x0, xzr               // encoding: [0x14,0x00,0x1f,0xba]

        sbc w29, w27, w25
        sbc wzr, w3, w4
        sbc w9, wzr, w10
        sbc w20, w0, wzr
// CHECK: sbc      w29, w27, w25              // encoding: [0x7d,0x03,0x19,0x5a]
// CHECK: sbc      wzr, w3, w4                // encoding: [0x7f,0x00,0x04,0x5a]
// CHECK: ngc      w9, w10                    // encoding: [0xe9,0x03,0x0a,0x5a]
// CHECK: sbc      w20, w0, wzr               // encoding: [0x14,0x00,0x1f,0x5a]

        sbc x29, x27, x25
        sbc xzr, x3, x4
        sbc x9, xzr, x10
        sbc x20, x0, xzr
// CHECK: sbc      x29, x27, x25              // encoding: [0x7d,0x03,0x19,0xda]
// CHECK: sbc      xzr, x3, x4                // encoding: [0x7f,0x00,0x04,0xda]
// CHECK: ngc      x9, x10                    // encoding: [0xe9,0x03,0x0a,0xda]
// CHECK: sbc      x20, x0, xzr               // encoding: [0x14,0x00,0x1f,0xda]

        sbcs w29, w27, w25
        sbcs wzr, w3, w4
        sbcs w9, wzr, w10
        sbcs w20, w0, wzr
// CHECK: sbcs     w29, w27, w25              // encoding: [0x7d,0x03,0x19,0x7a]
// CHECK: sbcs     wzr, w3, w4                // encoding: [0x7f,0x00,0x04,0x7a]
// CHECK: ngcs     w9, w10                    // encoding: [0xe9,0x03,0x0a,0x7a]
// CHECK: sbcs     w20, w0, wzr               // encoding: [0x14,0x00,0x1f,0x7a]

        sbcs x29, x27, x25
        sbcs xzr, x3, x4
        sbcs x9, xzr, x10
        sbcs x20, x0, xzr
// CHECK: sbcs     x29, x27, x25              // encoding: [0x7d,0x03,0x19,0xfa]
// CHECK: sbcs     xzr, x3, x4                // encoding: [0x7f,0x00,0x04,0xfa]
// CHECK: ngcs     x9, x10                    // encoding: [0xe9,0x03,0x0a,0xfa]
// CHECK: sbcs     x20, x0, xzr               // encoding: [0x14,0x00,0x1f,0xfa]

        ngc w3, w12
        ngc wzr, w9
        ngc w23, wzr
// CHECK: ngc      w3, w12                    // encoding: [0xe3,0x03,0x0c,0x5a]
// CHECK: ngc      wzr, w9                    // encoding: [0xff,0x03,0x09,0x5a]
// CHECK: ngc      w23, wzr                   // encoding: [0xf7,0x03,0x1f,0x5a]

        ngc x29, x30
        ngc xzr, x0
        ngc x0, xzr
// CHECK: ngc      x29, x30                   // encoding: [0xfd,0x03,0x1e,0xda]
// CHECK: ngc      xzr, x0                    // encoding: [0xff,0x03,0x00,0xda]
// CHECK: ngc      x0, xzr                    // encoding: [0xe0,0x03,0x1f,0xda]

        ngcs w3, w12
        ngcs wzr, w9
        ngcs w23, wzr
// CHECK: ngcs     w3, w12                    // encoding: [0xe3,0x03,0x0c,0x7a]
// CHECK: ngcs     wzr, w9                    // encoding: [0xff,0x03,0x09,0x7a]
// CHECK: ngcs     w23, wzr                   // encoding: [0xf7,0x03,0x1f,0x7a]

        ngcs x29, x30
        ngcs xzr, x0
        ngcs x0, xzr
// CHECK: ngcs     x29, x30                   // encoding: [0xfd,0x03,0x1e,0xfa]
// CHECK: ngcs     xzr, x0                    // encoding: [0xff,0x03,0x00,0xfa]
// CHECK: ngcs     x0, xzr                    // encoding: [0xe0,0x03,0x1f,0xfa]

//------------------------------------------------------------------------------
// Bitfield
//------------------------------------------------------------------------------

        sbfm x1, x2, #3, #4
        sbfm x3, x4, #63, #63
        sbfm wzr, wzr, #31, #31
        sbfm w12, w9, #0, #0

// CHECK: sbfx     x1, x2, #3, #2       // encoding: [0x41,0x10,0x43,0x93]
// CHECK: asr      x3, x4, #63          // encoding: [0x83,0xfc,0x7f,0x93]
// CHECK: asr      wzr, wzr, #31        // encoding: [0xff,0x7f,0x1f,0x13]
// CHECK: sbfx     w12, w9, #0, #1      // encoding: [0x2c,0x01,0x00,0x13]

        ubfm x4, x5, #12, #10
        ubfm xzr, x4, #0, #0
        ubfm x4, xzr, #63, #5
        ubfm x5, x6, #12, #63
// CHECK: ubfiz    x4, x5, #52, #11        // encoding: [0xa4,0x28,0x4c,0xd3]
// CHECK: ubfx     xzr, x4, #0, #1         // encoding: [0x9f,0x00,0x40,0xd3]
// CHECK: ubfiz    x4, xzr, #1, #6         // encoding: [0xe4,0x17,0x7f,0xd3]
// CHECK: lsr      x5, x6, #12             // encoding: [0xc5,0xfc,0x4c,0xd3]

        bfm x4, x5, #12, #10
        bfm xzr, x4, #0, #0
        bfm x4, xzr, #63, #5
        bfm x5, x6, #12, #63
// CHECK: bfi      x4, x5, #52, #11           // encoding: [0xa4,0x28,0x4c,0xb3]
// CHECK: bfxil    xzr, x4, #0, #1            // encoding: [0x9f,0x00,0x40,0xb3]
// CHECK: bfi      x4, xzr, #1, #6            // encoding: [0xe4,0x17,0x7f,0xb3]
// CHECK: bfxil    x5, x6, #12, #52           // encoding: [0xc5,0xfc,0x4c,0xb3]

        sxtb w1, w2
        sxtb xzr, w3
        sxth w9, w10
        sxth x0, w1
        sxtw x3, w30
// CHECK: sxtb     w1, w2                     // encoding: [0x41,0x1c,0x00,0x13]
// CHECK: sxtb     xzr, w3                    // encoding: [0x7f,0x1c,0x40,0x93]
// CHECK: sxth     w9, w10                    // encoding: [0x49,0x3d,0x00,0x13]
// CHECK: sxth     x0, w1                     // encoding: [0x20,0x3c,0x40,0x93]
// CHECK: sxtw     x3, w30                    // encoding: [0xc3,0x7f,0x40,0x93]

        uxtb w1, w2
        uxtb xzr, w3
        uxth w9, w10
        uxth x0, w1
// CHECK: uxtb     w1, w2                     // encoding: [0x41,0x1c,0x00,0x53]
// CHECK: uxtb     {{[wx]}}zr, w3             // encoding: [0x7f,0x1c,0x00,0x53]
// CHECK: uxth     w9, w10                    // encoding: [0x49,0x3d,0x00,0x53]
// CHECK: uxth     {{[wx]}}0, w1              // encoding: [0x20,0x3c,0x00,0x53]

        asr w3, w2, #0
        asr w9, w10, #31
        asr x20, x21, #63
        asr w1, wzr, #3
// CHECK: asr      w3, w2, #0                 // encoding: [0x43,0x7c,0x00,0x13]
// CHECK: asr      w9, w10, #31               // encoding: [0x49,0x7d,0x1f,0x13]
// CHECK: asr      x20, x21, #63              // encoding: [0xb4,0xfe,0x7f,0x93]
// CHECK: asr      w1, wzr, #3                // encoding: [0xe1,0x7f,0x03,0x13]

        lsr w3, w2, #0
        lsr w9, w10, #31
        lsr x20, x21, #63
        lsr wzr, wzr, #3
// CHECK: lsr      w3, w2, #0                 // encoding: [0x43,0x7c,0x00,0x53]
// CHECK: lsr      w9, w10, #31               // encoding: [0x49,0x7d,0x1f,0x53]
// CHECK: lsr      x20, x21, #63              // encoding: [0xb4,0xfe,0x7f,0xd3]
// CHECK: lsr      wzr, wzr, #3               // encoding: [0xff,0x7f,0x03,0x53]

        lsl w3, w2, #0
        lsl w9, w10, #31
        lsl x20, x21, #63
        lsl w1, wzr, #3
// CHECK: {{lsl|lsr}}      w3, w2, #0         // encoding: [0x43,0x7c,0x00,0x53]
// CHECK: lsl      w9, w10, #31               // encoding: [0x49,0x01,0x01,0x53]
// CHECK: lsl      x20, x21, #63              // encoding: [0xb4,0x02,0x41,0xd3]
// CHECK: lsl      w1, wzr, #3                // encoding: [0xe1,0x73,0x1d,0x53]

        sbfiz w9, w10, #0, #1
        sbfiz x2, x3, #63, #1
        sbfiz x19, x20, #0, #64
        sbfiz x9, x10, #5, #59
        sbfiz w9, w10, #0, #32
        sbfiz w11, w12, #31, #1
        sbfiz w13, w14, #29, #3
        sbfiz xzr, xzr, #10, #11
// CHECK: {{sbfiz|sbfx}}    w9, w10, #0, #1   // encoding: [0x49,0x01,0x00,0x13]
// CHECK: sbfiz    x2, x3, #63, #1            // encoding: [0x62,0x00,0x41,0x93]
// CHECK: asr    x19, x20, #0           // encoding: [0x93,0xfe,0x40,0x93]
// CHECK: sbfiz    x9, x10, #5, #59           // encoding: [0x49,0xe9,0x7b,0x93]
// CHECK: asr    w9, w10, #0            // encoding: [0x49,0x7d,0x00,0x13]
// CHECK: sbfiz    w11, w12, #31, #1          // encoding: [0x8b,0x01,0x01,0x13]
// CHECK: sbfiz    w13, w14, #29, #3          // encoding: [0xcd,0x09,0x03,0x13]
// CHECK: sbfiz    xzr, xzr, #10, #11         // encoding: [0xff,0x2b,0x76,0x93]

        sbfx w9, w10, #0, #1
        sbfx x2, x3, #63, #1
        sbfx x19, x20, #0, #64
        sbfx x9, x10, #5, #59
        sbfx w9, w10, #0, #32
        sbfx w11, w12, #31, #1
        sbfx w13, w14, #29, #3
        sbfx xzr, xzr, #10, #11
// CHECK: sbfx     w9, w10, #0, #1            // encoding: [0x49,0x01,0x00,0x13]
// CHECK: asr     x2, x3, #63           // encoding: [0x62,0xfc,0x7f,0x93]
// CHECK: asr     x19, x20, #0          // encoding: [0x93,0xfe,0x40,0x93]
// CHECK: asr     x9, x10, #5           // encoding: [0x49,0xfd,0x45,0x93]
// CHECK: asr     w9, w10, #0           // encoding: [0x49,0x7d,0x00,0x13]
// CHECK: asr     w11, w12, #31         // encoding: [0x8b,0x7d,0x1f,0x13]
// CHECK: asr     w13, w14, #29         // encoding: [0xcd,0x7d,0x1d,0x13]
// CHECK: sbfx     xzr, xzr, #10, #11         // encoding: [0xff,0x53,0x4a,0x93]

        bfi w9, w10, #0, #1
        bfi x2, x3, #63, #1
        bfi x19, x20, #0, #64
        bfi x9, x10, #5, #59
        bfi w9, w10, #0, #32
        bfi w11, w12, #31, #1
        bfi w13, w14, #29, #3
        bfi xzr, xzr, #10, #11

// CHECK: bfxil    w9, w10, #0, #1            // encoding: [0x49,0x01,0x00,0x33]
// CHECK: bfi      x2, x3, #63, #1            // encoding: [0x62,0x00,0x41,0xb3]
// CHECK: bfxil    x19, x20, #0, #64          // encoding: [0x93,0xfe,0x40,0xb3]
// CHECK: bfi      x9, x10, #5, #59           // encoding: [0x49,0xe9,0x7b,0xb3]
// CHECK: bfxil    w9, w10, #0, #32           // encoding: [0x49,0x7d,0x00,0x33]
// CHECK: bfi      w11, w12, #31, #1          // encoding: [0x8b,0x01,0x01,0x33]
// CHECK: bfi      w13, w14, #29, #3          // encoding: [0xcd,0x09,0x03,0x33]
// CHECK: bfi      xzr, xzr, #10, #11         // encoding: [0xff,0x2b,0x76,0xb3]

        bfxil w9, w10, #0, #1
        bfxil x2, x3, #63, #1
        bfxil x19, x20, #0, #64
        bfxil x9, x10, #5, #59
        bfxil w9, w10, #0, #32
        bfxil w11, w12, #31, #1
        bfxil w13, w14, #29, #3
        bfxil xzr, xzr, #10, #11
// CHECK: bfxil    w9, w10, #0, #1            // encoding: [0x49,0x01,0x00,0x33]
// CHECK: bfxil    x2, x3, #63, #1            // encoding: [0x62,0xfc,0x7f,0xb3]
// CHECK: bfxil    x19, x20, #0, #64          // encoding: [0x93,0xfe,0x40,0xb3]
// CHECK: bfxil    x9, x10, #5, #59           // encoding: [0x49,0xfd,0x45,0xb3]
// CHECK: bfxil    w9, w10, #0, #32           // encoding: [0x49,0x7d,0x00,0x33]
// CHECK: bfxil    w11, w12, #31, #1          // encoding: [0x8b,0x7d,0x1f,0x33]
// CHECK: bfxil    w13, w14, #29, #3          // encoding: [0xcd,0x7d,0x1d,0x33]
// CHECK: bfxil    xzr, xzr, #10, #11         // encoding: [0xff,0x53,0x4a,0xb3]

        ubfiz w9, w10, #0, #1
        ubfiz x2, x3, #63, #1
        ubfiz x19, x20, #0, #64
        ubfiz x9, x10, #5, #59
        ubfiz w9, w10, #0, #32
        ubfiz w11, w12, #31, #1
        ubfiz w13, w14, #29, #3
        ubfiz xzr, xzr, #10, #11

// CHECK: ubfx     w9, w10, #0, #1         // encoding: [0x49,0x01,0x00,0x53]
// CHECK: lsl      x2, x3, #63             // encoding: [0x62,0x00,0x41,0xd3]
// CHECK: lsr      x19, x20, #0            // encoding: [0x93,0xfe,0x40,0xd3]
// CHECK: lsl      x9, x10, #5             // encoding: [0x49,0xe9,0x7b,0xd3]
// CHECK: lsr      w9, w10, #0             // encoding: [0x49,0x7d,0x00,0x53]
// CHECK: lsl      w11, w12, #31           // encoding: [0x8b,0x01,0x01,0x53]
// CHECK: lsl      w13, w14, #29           // encoding: [0xcd,0x09,0x03,0x53]
// CHECK: ubfiz    xzr, xzr, #10, #11      // encoding: [0xff,0x2b,0x76,0xd3]

        ubfx w9, w10, #0, #1
        ubfx x2, x3, #63, #1
        ubfx x19, x20, #0, #64
        ubfx x9, x10, #5, #59
        ubfx w9, w10, #0, #32
        ubfx w11, w12, #31, #1
        ubfx w13, w14, #29, #3
        ubfx xzr, xzr, #10, #11

// CHECK: ubfx    w9, w10, #0, #1         // encoding: [0x49,0x01,0x00,0x53]
// CHECK: lsr     x2, x3, #63             // encoding: [0x62,0xfc,0x7f,0xd3]
// CHECK: lsr     x19, x20, #0            // encoding: [0x93,0xfe,0x40,0xd3]
// CHECK: lsr     x9, x10, #5             // encoding: [0x49,0xfd,0x45,0xd3]
// CHECK: lsr     w9, w10, #0             // encoding: [0x49,0x7d,0x00,0x53]
// CHECK: lsr     w11, w12, #31           // encoding: [0x8b,0x7d,0x1f,0x53]
// CHECK: lsr     w13, w14, #29           // encoding: [0xcd,0x7d,0x1d,0x53]
// CHECK: ubfx    xzr, xzr, #10, #11      // encoding: [0xff,0x53,0x4a,0xd3]

        bfc w3, #0, #32
        bfc wzr, #31, #1
        bfc x0, #5, #9
        bfc xzr, #63, #1
// CHECK: bfxil w3, wzr, #0, #32      // encoding: [0xe3,0x7f,0x00,0x33]
// CHECK: bfi wzr, wzr, #31, #1       // encoding: [0xff,0x03,0x01,0x33]
// CHECK: bfi x0, xzr, #5, #9         // encoding: [0xe0,0x23,0x7b,0xb3]
// CHECK: bfi xzr, xzr, #63, #1       // encoding: [0xff,0x03,0x41,0xb3]

//------------------------------------------------------------------------------
// Compare & branch (immediate)
//------------------------------------------------------------------------------

        cbz w5, lbl
        cbz x5, lbl
        cbnz x2, lbl
        cbnz x26, lbl
// CHECK: cbz    w5, lbl                 // encoding: [0bAAA00101,A,A,0x34]
// CHECK:                                 //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: cbz    x5, lbl                 // encoding: [0bAAA00101,A,A,0xb4]
// CHECK:                                 //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: cbnz    x2, lbl                 // encoding: [0bAAA00010,A,A,0xb5]
// CHECK:                                 //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: cbnz    x26, lbl                // encoding: [0bAAA11010,A,A,0xb5]
// CHECK:                                 //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        cbz wzr, lbl
        cbnz xzr, lbl

// CHECK: cbz    wzr, lbl                // encoding: [0bAAA11111,A,A,0x34]
// CHECK:                                 //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: cbnz    xzr, lbl                // encoding: [0bAAA11111,A,A,0xb5]
// CHECK:                                 //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        cbz w5, #0
        cbnz x3, #-4
        cbz w20, #1048572
        cbnz xzr, #-1048576
// CHECK: cbz     w5, #0                  // encoding: [0x05,0x00,0x00,0x34]
// CHECK: cbnz    x3, #-4                 // encoding: [0xe3,0xff,0xff,0xb5]
// CHECK: cbz     w20, #1048572           // encoding: [0xf4,0xff,0x7f,0x34]
// CHECK: cbnz    xzr, #-1048576          // encoding: [0x1f,0x00,0x80,0xb5]

//------------------------------------------------------------------------------
// Conditional branch (immediate)
//------------------------------------------------------------------------------

        b.eq lbl
        b.ne lbl
        b.cs lbl
        b.hs lbl
        b.lo lbl
        b.cc lbl
        b.mi lbl
        b.pl lbl
        b.vs lbl
        b.vc lbl
        b.hi lbl
        b.ls lbl
        b.ge lbl
        b.lt lbl
        b.gt lbl
        b.le lbl
        b.al lbl

// CHECK: b.eq lbl                     // encoding: [0bAAA00000,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.ne lbl                     // encoding: [0bAAA00001,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.hs lbl                     // encoding: [0bAAA00010,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.hs lbl                     // encoding: [0bAAA00010,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.lo lbl                     // encoding: [0bAAA00011,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.lo lbl                     // encoding: [0bAAA00011,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.mi lbl                     // encoding: [0bAAA00100,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.pl lbl                     // encoding: [0bAAA00101,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.vs lbl                     // encoding: [0bAAA00110,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.vc lbl                     // encoding: [0bAAA00111,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.hi lbl                     // encoding: [0bAAA01000,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.ls lbl                     // encoding: [0bAAA01001,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.ge lbl                     // encoding: [0bAAA01010,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.lt lbl                     // encoding: [0bAAA01011,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.gt lbl                     // encoding: [0bAAA01100,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.le lbl                     // encoding: [0bAAA01101,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: b.al lbl                     // encoding: [0bAAA01110,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        //  ARM64 has these in a separate file
        beq lbl
        bne lbl
        bcs lbl
        bhs lbl
        blo lbl
        bcc lbl
        bmi lbl
        bpl lbl
        bvs lbl
        bvc lbl
        bhi lbl
        bls lbl
        bge lbl
        blt lbl
        bgt lbl
        ble lbl
        bal lbl

        b.eq #0
        b.lt #-4
        b.cc #1048572
// CHECK: b.eq #0                         // encoding: [0x00,0x00,0x00,0x54]
// CHECK: b.lt #-4                        // encoding: [0xeb,0xff,0xff,0x54]
// CHECK: b.lo #1048572                   // encoding: [0xe3,0xff,0x7f,0x54]

//------------------------------------------------------------------------------
// Conditional compare (immediate)
//------------------------------------------------------------------------------

        ccmp w1, #31, #0, eq
        ccmp w3, #0, #15, hs
        ccmp wzr, #15, #13, cs
// CHECK: ccmp    w1, #31, #0, eq         // encoding: [0x20,0x08,0x5f,0x7a]
// CHECK: ccmp    w3, #0, #15, hs         // encoding: [0x6f,0x28,0x40,0x7a]
// CHECK: ccmp    wzr, #15, #13, hs       // encoding: [0xed,0x2b,0x4f,0x7a]

        ccmp x9, #31, #0, le
        ccmp x3, #0, #15, gt
        ccmp xzr, #5, #7, ne
// CHECK: ccmp    x9, #31, #0, le         // encoding: [0x20,0xd9,0x5f,0xfa]
// CHECK: ccmp    x3, #0, #15, gt         // encoding: [0x6f,0xc8,0x40,0xfa]
// CHECK: ccmp    xzr, #5, #7, ne         // encoding: [0xe7,0x1b,0x45,0xfa]

        ccmn w1, #31, #0, eq
        ccmn w3, #0, #15, hs
        ccmn wzr, #15, #13, cs
// CHECK: ccmn    w1, #31, #0, eq         // encoding: [0x20,0x08,0x5f,0x3a]
// CHECK: ccmn    w3, #0, #15, hs         // encoding: [0x6f,0x28,0x40,0x3a]
// CHECK: ccmn    wzr, #15, #13, hs       // encoding: [0xed,0x2b,0x4f,0x3a]

        ccmn x9, #31, #0, le
        ccmn x3, #0, #15, gt
        ccmn xzr, #5, #7, ne
// CHECK: ccmn    x9, #31, #0, le         // encoding: [0x20,0xd9,0x5f,0xba]
// CHECK: ccmn    x3, #0, #15, gt         // encoding: [0x6f,0xc8,0x40,0xba]
// CHECK: ccmn    xzr, #5, #7, ne         // encoding: [0xe7,0x1b,0x45,0xba]

//------------------------------------------------------------------------------
// Conditional compare (register)
//------------------------------------------------------------------------------

        ccmp w1, wzr, #0, eq
        ccmp w3, w0, #15, hs
        ccmp wzr, w15, #13, cs
// CHECK: ccmp    w1, wzr, #0, eq         // encoding: [0x20,0x00,0x5f,0x7a]
// CHECK: ccmp    w3, w0, #15, hs         // encoding: [0x6f,0x20,0x40,0x7a]
// CHECK: ccmp    wzr, w15, #13, hs       // encoding: [0xed,0x23,0x4f,0x7a]

        ccmp x9, xzr, #0, le
        ccmp x3, x0, #15, gt
        ccmp xzr, x5, #7, ne
// CHECK: ccmp    x9, xzr, #0, le         // encoding: [0x20,0xd1,0x5f,0xfa]
// CHECK: ccmp    x3, x0, #15, gt         // encoding: [0x6f,0xc0,0x40,0xfa]
// CHECK: ccmp    xzr, x5, #7, ne         // encoding: [0xe7,0x13,0x45,0xfa]

        ccmn w1, wzr, #0, eq
        ccmn w3, w0, #15, hs
        ccmn wzr, w15, #13, cs
// CHECK: ccmn    w1, wzr, #0, eq         // encoding: [0x20,0x00,0x5f,0x3a]
// CHECK: ccmn    w3, w0, #15, hs         // encoding: [0x6f,0x20,0x40,0x3a]
// CHECK: ccmn    wzr, w15, #13, hs       // encoding: [0xed,0x23,0x4f,0x3a]

        ccmn x9, xzr, #0, le
        ccmn x3, x0, #15, gt
        ccmn xzr, x5, #7, ne
// CHECK: ccmn    x9, xzr, #0, le         // encoding: [0x20,0xd1,0x5f,0xba]
// CHECK: ccmn    x3, x0, #15, gt         // encoding: [0x6f,0xc0,0x40,0xba]
// CHECK: ccmn    xzr, x5, #7, ne         // encoding: [0xe7,0x13,0x45,0xba]

//------------------------------------------------------------------------------
// Conditional select
//------------------------------------------------------------------------------
        csel w1, w0, w19, ne
        csel wzr, w5, w9, eq
        csel w9, wzr, w30, gt
        csel w1, w28, wzr, mi
// CHECK: csel     w1, w0, w19, ne            // encoding: [0x01,0x10,0x93,0x1a]
// CHECK: csel     wzr, w5, w9, eq            // encoding: [0xbf,0x00,0x89,0x1a]
// CHECK: csel     w9, wzr, w30, gt           // encoding: [0xe9,0xc3,0x9e,0x1a]
// CHECK: csel     w1, w28, wzr, mi           // encoding: [0x81,0x43,0x9f,0x1a]

        csel x19, x23, x29, lt
        csel xzr, x3, x4, ge
        csel x5, xzr, x6, cs
        csel x7, x8, xzr, cc
// CHECK: csel     x19, x23, x29, lt          // encoding: [0xf3,0xb2,0x9d,0x9a]
// CHECK: csel     xzr, x3, x4, ge            // encoding: [0x7f,0xa0,0x84,0x9a]
// CHECK: csel     x5, xzr, x6, hs            // encoding: [0xe5,0x23,0x86,0x9a]
// CHECK: csel     x7, x8, xzr, lo            // encoding: [0x07,0x31,0x9f,0x9a]

        csinc w1, w0, w19, ne
        csinc wzr, w5, w9, eq
        csinc w9, wzr, w30, gt
        csinc w1, w28, wzr, mi
// CHECK: csinc    w1, w0, w19, ne            // encoding: [0x01,0x14,0x93,0x1a]
// CHECK: csinc    wzr, w5, w9, eq            // encoding: [0xbf,0x04,0x89,0x1a]
// CHECK: csinc    w9, wzr, w30, gt           // encoding: [0xe9,0xc7,0x9e,0x1a]
// CHECK: csinc    w1, w28, wzr, mi           // encoding: [0x81,0x47,0x9f,0x1a]

        csinc x19, x23, x29, lt
        csinc xzr, x3, x4, ge
        csinc x5, xzr, x6, cs
        csinc x7, x8, xzr, cc
// CHECK: csinc    x19, x23, x29, lt          // encoding: [0xf3,0xb6,0x9d,0x9a]
// CHECK: csinc    xzr, x3, x4, ge            // encoding: [0x7f,0xa4,0x84,0x9a]
// CHECK: csinc    x5, xzr, x6, hs            // encoding: [0xe5,0x27,0x86,0x9a]
// CHECK: csinc    x7, x8, xzr, lo            // encoding: [0x07,0x35,0x9f,0x9a]

        csinv w1, w0, w19, ne
        csinv wzr, w5, w9, eq
        csinv w9, wzr, w30, gt
        csinv w1, w28, wzr, mi
// CHECK: csinv    w1, w0, w19, ne            // encoding: [0x01,0x10,0x93,0x5a]
// CHECK: csinv    wzr, w5, w9, eq            // encoding: [0xbf,0x00,0x89,0x5a]
// CHECK: csinv    w9, wzr, w30, gt           // encoding: [0xe9,0xc3,0x9e,0x5a]
// CHECK: csinv    w1, w28, wzr, mi           // encoding: [0x81,0x43,0x9f,0x5a]

        csinv x19, x23, x29, lt
        csinv xzr, x3, x4, ge
        csinv x5, xzr, x6, cs
        csinv x7, x8, xzr, cc
// CHECK: csinv    x19, x23, x29, lt          // encoding: [0xf3,0xb2,0x9d,0xda]
// CHECK: csinv    xzr, x3, x4, ge            // encoding: [0x7f,0xa0,0x84,0xda]
// CHECK: csinv    x5, xzr, x6, hs            // encoding: [0xe5,0x23,0x86,0xda]
// CHECK: csinv    x7, x8, xzr, lo            // encoding: [0x07,0x31,0x9f,0xda]

        csneg w1, w0, w19, ne
        csneg wzr, w5, w9, eq
        csneg w9, wzr, w30, gt
        csneg w1, w28, wzr, mi
// CHECK: csneg    w1, w0, w19, ne            // encoding: [0x01,0x14,0x93,0x5a]
// CHECK: csneg    wzr, w5, w9, eq            // encoding: [0xbf,0x04,0x89,0x5a]
// CHECK: csneg    w9, wzr, w30, gt           // encoding: [0xe9,0xc7,0x9e,0x5a]
// CHECK: csneg    w1, w28, wzr, mi           // encoding: [0x81,0x47,0x9f,0x5a]

        csneg x19, x23, x29, lt
        csneg xzr, x3, x4, ge
        csneg x5, xzr, x6, cs
        csneg x7, x8, xzr, cc
// CHECK: csneg    x19, x23, x29, lt          // encoding: [0xf3,0xb6,0x9d,0xda]
// CHECK: csneg    xzr, x3, x4, ge            // encoding: [0x7f,0xa4,0x84,0xda]
// CHECK: csneg    x5, xzr, x6, hs            // encoding: [0xe5,0x27,0x86,0xda]
// CHECK: csneg    x7, x8, xzr, lo            // encoding: [0x07,0x35,0x9f,0xda]

        cset w3, eq
        cset x9, pl
// CHECK: cset    w3, eq           // encoding: [0xe3,0x17,0x9f,0x1a]
// CHECK: cset    x9, pl           // encoding: [0xe9,0x47,0x9f,0x9a]

        csetm w20, ne
        csetm x30, ge
// CHECK: csetm    w20, ne          // encoding: [0xf4,0x03,0x9f,0x5a]
// CHECK: csetm    x30, ge          // encoding: [0xfe,0xb3,0x9f,0xda]

        cinc w3, w5, gt
        cinc wzr, w4, le
        cinc w9, wzr, lt
// CHECK: cinc    w3, w5, gt             // encoding: [0xa3,0xd4,0x85,0x1a]
// CHECK: cinc    wzr, w4, le            // encoding: [0x9f,0xc4,0x84,0x1a]
// CHECK: cset    w9, lt           // encoding: [0xe9,0xa7,0x9f,0x1a]

        cinc x3, x5, gt
        cinc xzr, x4, le
        cinc x9, xzr, lt
// CHECK: cinc     x3, x5, gt             // encoding: [0xa3,0xd4,0x85,0x9a]
// CHECK: cinc     xzr, x4, le            // encoding: [0x9f,0xc4,0x84,0x9a]
// CHECK: cset     x9, lt           // encoding: [0xe9,0xa7,0x9f,0x9a]

        cinv w3, w5, gt
        cinv wzr, w4, le
        cinv w9, wzr, lt
// CHECK: cinv    w3, w5, gt             // encoding: [0xa3,0xd0,0x85,0x5a]
// CHECK: cinv    wzr, w4, le            // encoding: [0x9f,0xc0,0x84,0x5a]
// CHECK: csetm    w9, lt           // encoding: [0xe9,0xa3,0x9f,0x5a]

        cinv x3, x5, gt
        cinv xzr, x4, le
        cinv x9, xzr, lt
// CHECK: cinv    x3, x5, gt             // encoding: [0xa3,0xd0,0x85,0xda]
// CHECK: cinv    xzr, x4, le            // encoding: [0x9f,0xc0,0x84,0xda]
// CHECK: csetm    x9, lt           // encoding: [0xe9,0xa3,0x9f,0xda]

        cneg w3, w5, gt
        cneg wzr, w4, le
        cneg w9, wzr, lt
// CHECK: cneg    w3, w5, gt            // encoding: [0xa3,0xd4,0x85,0x5a]
// CHECK: cneg    wzr, w4, le            // encoding: [0x9f,0xc4,0x84,0x5a]
// CHECK: cneg    w9, wzr, lt           // encoding: [0xe9,0xa7,0x9f,0x5a]

        cneg x3, x5, gt
        cneg xzr, x4, le
        cneg x9, xzr, lt
// CHECK: cneg    x3, x5, gt             // encoding: [0xa3,0xd4,0x85,0xda]
// CHECK: cneg    xzr, x4, le            // encoding: [0x9f,0xc4,0x84,0xda]
// CHECK: cneg    x9, xzr, lt           // encoding: [0xe9,0xa7,0x9f,0xda]

//------------------------------------------------------------------------------
// Data-processing (1 source)
//------------------------------------------------------------------------------

	rbit	w0, w7
	rbit	x18, x3
	rev16	w17, w1
	rev16	x5, x2
	rev	w18, w0
	rev32	x20, x1
	rev32	x20, xzr
// CHECK: rbit	w0, w7                       // encoding: [0xe0,0x00,0xc0,0x5a]
// CHECK: rbit	x18, x3                      // encoding: [0x72,0x00,0xc0,0xda]
// CHECK: rev16 w17, w1                      // encoding: [0x31,0x04,0xc0,0x5a]
// CHECK: rev16	x5, x2                       // encoding: [0x45,0x04,0xc0,0xda]
// CHECK: rev	w18, w0                      // encoding: [0x12,0x08,0xc0,0x5a]
// CHECK: rev32	x20, x1                      // encoding: [0x34,0x08,0xc0,0xda]
// CHECK: rev32	x20, xzr                     // encoding: [0xf4,0x0b,0xc0,0xda]

	rev	x22, x2
	rev	x18, xzr
	rev	w7, wzr
	clz	w24, w3
	clz	x26, x4
	cls	w3, w5
	cls	x20, x5
// CHECK: rev	x22, x2                      // encoding: [0x56,0x0c,0xc0,0xda]
// CHECK: rev	x18, xzr                     // encoding: [0xf2,0x0f,0xc0,0xda]
// CHECK: rev	w7, wzr                      // encoding: [0xe7,0x0b,0xc0,0x5a]
// CHECK: clz	w24, w3                      // encoding: [0x78,0x10,0xc0,0x5a]
// CHECK: clz	x26, x4                      // encoding: [0x9a,0x10,0xc0,0xda]
// CHECK: cls	w3, w5                       // encoding: [0xa3,0x14,0xc0,0x5a]
// CHECK: cls	x20, x5                      // encoding: [0xb4,0x14,0xc0,0xda]

	clz	w24, wzr
	rev	x22, xzr
// CHECK: clz	w24, wzr                     // encoding: [0xf8,0x13,0xc0,0x5a]
// CHECK: rev	x22, xzr                     // encoding: [0xf6,0x0f,0xc0,0xda]

	rev64	x13, x12
// CHECK: rev	x13, x12                     // encoding: [0x8d,0x0d,0xc0,0xda]

//------------------------------------------------------------------------------
// Data-processing (2 source)
//------------------------------------------------------------------------------

        udiv	w0, w7, w10
        udiv	x9, x22, x4
        sdiv	w12, w21, w0
        sdiv	x13, x2, x1
        lslv	w11, w12, w13
        lslv	x14, x15, x16
        lsrv	w17, w18, w19
        lsrv	x20, x21, x22
        asrv	w23, w24, w25
        asrv	x26, x27, x28
        rorv	w0, w1, w2
        rorv    x3, x4, x5


// CHECK: udiv	w0, w7, w10                   // encoding: [0xe0,0x08,0xca,0x1a]
// CHECK: udiv	x9, x22, x4                   // encoding: [0xc9,0x0a,0xc4,0x9a]
// CHECK: sdiv	w12, w21, w0                  // encoding: [0xac,0x0e,0xc0,0x1a]
// CHECK: sdiv	x13, x2, x1                   // encoding: [0x4d,0x0c,0xc1,0x9a]
// CHECK: lsl	w11, w12, w13                 // encoding: [0x8b,0x21,0xcd,0x1a]
// CHECK: lsl	x14, x15, x16                 // encoding: [0xee,0x21,0xd0,0x9a]
// CHECK: lsr	w17, w18, w19                 // encoding: [0x51,0x26,0xd3,0x1a]
// CHECK: lsr	x20, x21, x22                 // encoding: [0xb4,0x26,0xd6,0x9a]
// CHECK: asr	w23, w24, w25                 // encoding: [0x17,0x2b,0xd9,0x1a]
// CHECK: asr	x26, x27, x28                 // encoding: [0x7a,0x2b,0xdc,0x9a]
// CHECK: ror	w0, w1, w2                    // encoding: [0x20,0x2c,0xc2,0x1a]
// CHECK: ror  x3, x4, x5                     // encoding: [0x83,0x2c,0xc5,0x9a]


        lsl	w6, w7, w8
        lsl	x9, x10, x11
        lsr	w12, w13, w14
        lsr	x15, x16, x17
        asr	w18, w19, w20
        asr	x21, x22, x23
        ror	w24, w25, w26
        ror	x27, x28, x29
// CHECK: lsl	w6, w7, w8                    // encoding: [0xe6,0x20,0xc8,0x1a]
// CHECK: lsl	x9, x10, x11                  // encoding: [0x49,0x21,0xcb,0x9a]
// CHECK: lsr	w12, w13, w14                 // encoding: [0xac,0x25,0xce,0x1a]
// CHECK: lsr	x15, x16, x17                 // encoding: [0x0f,0x26,0xd1,0x9a]
// CHECK: asr	w18, w19, w20                 // encoding: [0x72,0x2a,0xd4,0x1a]
// CHECK: asr	x21, x22, x23                 // encoding: [0xd5,0x2a,0xd7,0x9a]
// CHECK: ror	w24, w25, w26                 // encoding: [0x38,0x2f,0xda,0x1a]
// CHECK: ror	x27, x28, x29                 // encoding: [0x9b,0x2f,0xdd,0x9a]

        madd w1, w3, w7, w4
        madd wzr, w0, w9, w11
        madd w13, wzr, w4, w4
        madd w19, w30, wzr, w29
        madd w4, w5, w6, wzr
// CHECK: madd     w1, w3, w7, w4             // encoding: [0x61,0x10,0x07,0x1b]
// CHECK: madd     wzr, w0, w9, w11           // encoding: [0x1f,0x2c,0x09,0x1b]
// CHECK: madd     w13, wzr, w4, w4           // encoding: [0xed,0x13,0x04,0x1b]
// CHECK: madd     w19, w30, wzr, w29         // encoding: [0xd3,0x77,0x1f,0x1b]
// CHECK: mul      w4, w5, w6                 // encoding: [0xa4,0x7c,0x06,0x1b]

        madd x1, x3, x7, x4
        madd xzr, x0, x9, x11
        madd x13, xzr, x4, x4
        madd x19, x30, xzr, x29
        madd x4, x5, x6, xzr
// CHECK: madd     x1, x3, x7, x4             // encoding: [0x61,0x10,0x07,0x9b]
// CHECK: madd     xzr, x0, x9, x11           // encoding: [0x1f,0x2c,0x09,0x9b]
// CHECK: madd     x13, xzr, x4, x4           // encoding: [0xed,0x13,0x04,0x9b]
// CHECK: madd     x19, x30, xzr, x29         // encoding: [0xd3,0x77,0x1f,0x9b]
// CHECK: mul      x4, x5, x6                 // encoding: [0xa4,0x7c,0x06,0x9b]

        msub w1, w3, w7, w4
        msub wzr, w0, w9, w11
        msub w13, wzr, w4, w4
        msub w19, w30, wzr, w29
        msub w4, w5, w6, wzr
// CHECK: msub     w1, w3, w7, w4             // encoding: [0x61,0x90,0x07,0x1b]
// CHECK: msub     wzr, w0, w9, w11           // encoding: [0x1f,0xac,0x09,0x1b]
// CHECK: msub     w13, wzr, w4, w4           // encoding: [0xed,0x93,0x04,0x1b]
// CHECK: msub     w19, w30, wzr, w29         // encoding: [0xd3,0xf7,0x1f,0x1b]
// CHECK: mneg     w4, w5, w6                 // encoding: [0xa4,0xfc,0x06,0x1b]

        msub x1, x3, x7, x4
        msub xzr, x0, x9, x11
        msub x13, xzr, x4, x4
        msub x19, x30, xzr, x29
        msub x4, x5, x6, xzr
// CHECK: msub     x1, x3, x7, x4             // encoding: [0x61,0x90,0x07,0x9b]
// CHECK: msub     xzr, x0, x9, x11           // encoding: [0x1f,0xac,0x09,0x9b]
// CHECK: msub     x13, xzr, x4, x4           // encoding: [0xed,0x93,0x04,0x9b]
// CHECK: msub     x19, x30, xzr, x29         // encoding: [0xd3,0xf7,0x1f,0x9b]
// CHECK: mneg     x4, x5, x6                 // encoding: [0xa4,0xfc,0x06,0x9b]

        smaddl x3, w5, w2, x9
        smaddl xzr, w10, w11, x12
        smaddl x13, wzr, w14, x15
        smaddl x16, w17, wzr, x18
        smaddl x19, w20, w21, xzr
// CHECK: smaddl   x3, w5, w2, x9             // encoding: [0xa3,0x24,0x22,0x9b]
// CHECK: smaddl   xzr, w10, w11, x12         // encoding: [0x5f,0x31,0x2b,0x9b]
// CHECK: smaddl   x13, wzr, w14, x15         // encoding: [0xed,0x3f,0x2e,0x9b]
// CHECK: smaddl   x16, w17, wzr, x18         // encoding: [0x30,0x4a,0x3f,0x9b]
// CHECK: smull    x19, w20, w21              // encoding: [0x93,0x7e,0x35,0x9b]

        smsubl x3, w5, w2, x9
        smsubl xzr, w10, w11, x12
        smsubl x13, wzr, w14, x15
        smsubl x16, w17, wzr, x18
        smsubl x19, w20, w21, xzr
// CHECK: smsubl   x3, w5, w2, x9             // encoding: [0xa3,0xa4,0x22,0x9b]
// CHECK: smsubl   xzr, w10, w11, x12         // encoding: [0x5f,0xb1,0x2b,0x9b]
// CHECK: smsubl   x13, wzr, w14, x15         // encoding: [0xed,0xbf,0x2e,0x9b]
// CHECK: smsubl   x16, w17, wzr, x18         // encoding: [0x30,0xca,0x3f,0x9b]
// CHECK: smnegl   x19, w20, w21              // encoding: [0x93,0xfe,0x35,0x9b]

        umaddl x3, w5, w2, x9
        umaddl xzr, w10, w11, x12
        umaddl x13, wzr, w14, x15
        umaddl x16, w17, wzr, x18
        umaddl x19, w20, w21, xzr
// CHECK: umaddl   x3, w5, w2, x9             // encoding: [0xa3,0x24,0xa2,0x9b]
// CHECK: umaddl   xzr, w10, w11, x12         // encoding: [0x5f,0x31,0xab,0x9b]
// CHECK: umaddl   x13, wzr, w14, x15         // encoding: [0xed,0x3f,0xae,0x9b]
// CHECK: umaddl   x16, w17, wzr, x18         // encoding: [0x30,0x4a,0xbf,0x9b]
// CHECK: umull    x19, w20, w21              // encoding: [0x93,0x7e,0xb5,0x9b]



        umsubl x3, w5, w2, x9
        umsubl xzr, w10, w11, x12
        umsubl x13, wzr, w14, x15
        umsubl x16, w17, wzr, x18
        umsubl x19, w20, w21, xzr
// CHECK: umsubl   x3, w5, w2, x9             // encoding: [0xa3,0xa4,0xa2,0x9b]
// CHECK: umsubl   xzr, w10, w11, x12         // encoding: [0x5f,0xb1,0xab,0x9b]
// CHECK: umsubl   x13, wzr, w14, x15         // encoding: [0xed,0xbf,0xae,0x9b]
// CHECK: umsubl   x16, w17, wzr, x18         // encoding: [0x30,0xca,0xbf,0x9b]
// CHECK: umnegl   x19, w20, w21              // encoding: [0x93,0xfe,0xb5,0x9b]

        smulh x30, x29, x28
        smulh xzr, x27, x26
        smulh x25, xzr, x24
        smulh x23, x22, xzr
// CHECK: smulh    x30, x29, x28              // encoding: [0xbe,0x7f,0x5c,0x9b]
// CHECK: smulh    xzr, x27, x26              // encoding: [0x7f,0x7f,0x5a,0x9b]
// CHECK: smulh    x25, xzr, x24              // encoding: [0xf9,0x7f,0x58,0x9b]
// CHECK: smulh    x23, x22, xzr              // encoding: [0xd7,0x7e,0x5f,0x9b]

        umulh x30, x29, x28
        umulh xzr, x27, x26
        umulh x25, xzr, x24
        umulh x23, x22, xzr
// CHECK: umulh    x30, x29, x28              // encoding: [0xbe,0x7f,0xdc,0x9b]
// CHECK: umulh    xzr, x27, x26              // encoding: [0x7f,0x7f,0xda,0x9b]
// CHECK: umulh    x25, xzr, x24              // encoding: [0xf9,0x7f,0xd8,0x9b]
// CHECK: umulh    x23, x22, xzr              // encoding: [0xd7,0x7e,0xdf,0x9b]

        mul w3, w4, w5
        mul wzr, w6, w7
        mul w8, wzr, w9
        mul w10, w11, wzr

        mul x12, x13, x14
        mul xzr, x15, x16
        mul x17, xzr, x18
        mul x19, x20, xzr

        mneg w21, w22, w23
        mneg wzr, w24, w25
        mneg w26, wzr, w27
        mneg w28, w29, wzr

        smull x11, w13, w17
        umull x11, w13, w17
        smnegl x11, w13, w17
        umnegl x11, w13, w17
// CHECK: mul      w3, w4, w5                 // encoding: [0x83,0x7c,0x05,0x1b]
// CHECK: mul      wzr, w6, w7                // encoding: [0xdf,0x7c,0x07,0x1b]
// CHECK: mul      w8, wzr, w9                // encoding: [0xe8,0x7f,0x09,0x1b]
// CHECK: mul      w10, w11, wzr              // encoding: [0x6a,0x7d,0x1f,0x1b]
// CHECK: mul      x12, x13, x14              // encoding: [0xac,0x7d,0x0e,0x9b]
// CHECK: mul      xzr, x15, x16              // encoding: [0xff,0x7d,0x10,0x9b]
// CHECK: mul      x17, xzr, x18              // encoding: [0xf1,0x7f,0x12,0x9b]
// CHECK: mul      x19, x20, xzr              // encoding: [0x93,0x7e,0x1f,0x9b]
// CHECK: mneg     w21, w22, w23              // encoding: [0xd5,0xfe,0x17,0x1b]
// CHECK: mneg     wzr, w24, w25              // encoding: [0x1f,0xff,0x19,0x1b]
// CHECK: mneg     w26, wzr, w27              // encoding: [0xfa,0xff,0x1b,0x1b]
// CHECK: mneg     w28, w29, wzr              // encoding: [0xbc,0xff,0x1f,0x1b]
// CHECK: smull    x11, w13, w17              // encoding: [0xab,0x7d,0x31,0x9b]
// CHECK: umull    x11, w13, w17              // encoding: [0xab,0x7d,0xb1,0x9b]
// CHECK: smnegl   x11, w13, w17              // encoding: [0xab,0xfd,0x31,0x9b]
// CHECK: umnegl   x11, w13, w17              // encoding: [0xab,0xfd,0xb1,0x9b]

//------------------------------------------------------------------------------
// Exception generation
//------------------------------------------------------------------------------
        svc #0
        svc #65535
// CHECK: svc      #0                         // encoding: [0x01,0x00,0x00,0xd4]
// CHECK: svc      #{{65535|0xffff}}          // encoding: [0xe1,0xff,0x1f,0xd4]

        hvc #1
	smc #12000
        brk #12
        hlt #123
// CHECK: hvc      #{{1|0x1}}                 // encoding: [0x22,0x00,0x00,0xd4]
// CHECK: smc      #{{12000|0x2ee0}}          // encoding: [0x03,0xdc,0x05,0xd4]
// CHECK: brk      #{{12|0xc}}                // encoding: [0x80,0x01,0x20,0xd4]
// CHECK: hlt      #{{123|0x7b}}              // encoding: [0x60,0x0f,0x40,0xd4]

        dcps1 #42
        dcps2 #9
	dcps3 #1000
// CHECK: dcps1    #{{42|0x2a}}               // encoding: [0x41,0x05,0xa0,0xd4]
// CHECK: dcps2    #{{9|0x9}}                 // encoding: [0x22,0x01,0xa0,0xd4]
// CHECK: dcps3    #{{1000|0x3e8}}            // encoding: [0x03,0x7d,0xa0,0xd4]

        dcps1
        dcps2
	dcps3
// CHECK: dcps1                               // encoding: [0x01,0x00,0xa0,0xd4]
// CHECK: dcps2                               // encoding: [0x02,0x00,0xa0,0xd4]
// CHECK: dcps3                               // encoding: [0x03,0x00,0xa0,0xd4]

//------------------------------------------------------------------------------
// Extract (immediate)
//------------------------------------------------------------------------------

        extr w3, w5, w7, #0
        extr w11, w13, w17, #31
// CHECK: extr     w3, w5, w7, #0             // encoding: [0xa3,0x00,0x87,0x13]
// CHECK: extr     w11, w13, w17, #31         // encoding: [0xab,0x7d,0x91,0x13]

        extr x3, x5, x7, #15
        extr x11, x13, x17, #63
// CHECK: extr     x3, x5, x7, #15            // encoding: [0xa3,0x3c,0xc7,0x93]
// CHECK: extr     x11, x13, x17, #63         // encoding: [0xab,0xfd,0xd1,0x93]

        ror x19, x23, #24
        ror x29, xzr, #63
// CHECK: ror     x19, x23, #24         // encoding: [0xf3,0x62,0xd7,0x93]
// CHECK: ror     x29, xzr, #63         // encoding: [0xfd,0xff,0xdf,0x93]

        ror w9, w13, #31
// CHECK: ror     w9, w13, #31          // encoding: [0xa9,0x7d,0x8d,0x13]

//------------------------------------------------------------------------------
// Floating-point compare
//------------------------------------------------------------------------------

        fcmp s3, s5
        fcmp s31, #0.0
// CHECK: fcmp    s3, s5                  // encoding: [0x60,0x20,0x25,0x1e]
// CHECK: fcmp    s31, #0.0               // encoding: [0xe8,0x23,0x20,0x1e]

        fcmpe s29, s30
        fcmpe s15, #0.0
// CHECK: fcmpe   s29, s30                // encoding: [0xb0,0x23,0x3e,0x1e]
// CHECK: fcmpe   s15, #0.0               // encoding: [0xf8,0x21,0x20,0x1e]

        fcmp d4, d12
        fcmp d23, #0.0
// CHECK: fcmp    d4, d12                 // encoding: [0x80,0x20,0x6c,0x1e]
// CHECK: fcmp    d23, #0.0               // encoding: [0xe8,0x22,0x60,0x1e]

        fcmpe d26, d22
        fcmpe d29, #0.0
// CHECK: fcmpe   d26, d22                // encoding: [0x50,0x23,0x76,0x1e]
// CHECK: fcmpe   d29, #0.0               // encoding: [0xb8,0x23,0x60,0x1e]

//------------------------------------------------------------------------------
// Floating-point conditional compare
//------------------------------------------------------------------------------

        fccmp s1, s31, #0, eq
        fccmp s3, s0, #15, hs
        fccmp s31, s15, #13, cs
// CHECK: fccmp    s1, s31, #0, eq         // encoding: [0x20,0x04,0x3f,0x1e]
// CHECK: fccmp    s3, s0, #15, hs         // encoding: [0x6f,0x24,0x20,0x1e]
// CHECK: fccmp    s31, s15, #13, hs       // encoding: [0xed,0x27,0x2f,0x1e]

        fccmp d9, d31, #0, le
        fccmp d3, d0, #15, gt
        fccmp d31, d5, #7, ne
// CHECK: fccmp    d9, d31, #0, le         // encoding: [0x20,0xd5,0x7f,0x1e]
// CHECK: fccmp    d3, d0, #15, gt         // encoding: [0x6f,0xc4,0x60,0x1e]
// CHECK: fccmp    d31, d5, #7, ne         // encoding: [0xe7,0x17,0x65,0x1e]

        fccmpe s1, s31, #0, eq
        fccmpe s3, s0, #15, hs
        fccmpe s31, s15, #13, cs
// CHECK: fccmpe    s1, s31, #0, eq         // encoding: [0x30,0x04,0x3f,0x1e]
// CHECK: fccmpe    s3, s0, #15, hs         // encoding: [0x7f,0x24,0x20,0x1e]
// CHECK: fccmpe    s31, s15, #13, hs       // encoding: [0xfd,0x27,0x2f,0x1e]

        fccmpe d9, d31, #0, le
        fccmpe d3, d0, #15, gt
        fccmpe d31, d5, #7, ne
// CHECK: fccmpe    d9, d31, #0, le         // encoding: [0x30,0xd5,0x7f,0x1e]
// CHECK: fccmpe    d3, d0, #15, gt         // encoding: [0x7f,0xc4,0x60,0x1e]
// CHECK: fccmpe    d31, d5, #7, ne         // encoding: [0xf7,0x17,0x65,0x1e]

//------------------------------------------------------------------------------
// Floating-point conditional compare
//------------------------------------------------------------------------------

        fcsel s3, s20, s9, pl
        fcsel d9, d10, d11, mi
// CHECK: fcsel   s3, s20, s9, pl         // encoding: [0x83,0x5e,0x29,0x1e]
// CHECK: fcsel   d9, d10, d11, mi        // encoding: [0x49,0x4d,0x6b,0x1e]

//------------------------------------------------------------------------------
// Floating-point data-processing (1 source)
//------------------------------------------------------------------------------

        fmov s0, s1
        fabs s2, s3
        fneg s4, s5
        fsqrt s6, s7
        fcvt d8, s9
        fcvt h10, s11
        frintn s12, s13
        frintp s14, s15
        frintm s16, s17
        frintz s18, s19
        frinta s20, s21
        frintx s22, s23
        frinti s24, s25
// CHECK: fmov     s0, s1                // encoding: [0x20,0x40,0x20,0x1e]
// CHECK: fabs     s2, s3                // encoding: [0x62,0xc0,0x20,0x1e]
// CHECK: fneg     s4, s5                     // encoding: [0xa4,0x40,0x21,0x1e]
// CHECK: fsqrt    s6, s7                     // encoding: [0xe6,0xc0,0x21,0x1e]
// CHECK: fcvt     d8, s9                     // encoding: [0x28,0xc1,0x22,0x1e]
// CHECK: fcvt     h10, s11                   // encoding: [0x6a,0xc1,0x23,0x1e]
// CHECK: frintn   s12, s13                   // encoding: [0xac,0x41,0x24,0x1e]
// CHECK: frintp   s14, s15                   // encoding: [0xee,0xc1,0x24,0x1e]
// CHECK: frintm   s16, s17                   // encoding: [0x30,0x42,0x25,0x1e]
// CHECK: frintz   s18, s19                   // encoding: [0x72,0xc2,0x25,0x1e]
// CHECK: frinta   s20, s21                   // encoding: [0xb4,0x42,0x26,0x1e]
// CHECK: frintx   s22, s23                   // encoding: [0xf6,0x42,0x27,0x1e]
// CHECK: frinti   s24, s25                   // encoding: [0x38,0xc3,0x27,0x1e]

        fmov d0, d1
        fabs d2, d3
        fneg d4, d5
        fsqrt d6, d7
        fcvt s8, d9
        fcvt h10, d11
        frintn d12, d13
        frintp d14, d15
        frintm d16, d17
        frintz d18, d19
        frinta d20, d21
        frintx d22, d23
        frinti d24, d25
// CHECK: fmov     d0, d1                     // encoding: [0x20,0x40,0x60,0x1e]
// CHECK: fabs     d2, d3                     // encoding: [0x62,0xc0,0x60,0x1e]
// CHECK: fneg     d4, d5                     // encoding: [0xa4,0x40,0x61,0x1e]
// CHECK: fsqrt    d6, d7                     // encoding: [0xe6,0xc0,0x61,0x1e]
// CHECK: fcvt     s8, d9                     // encoding: [0x28,0x41,0x62,0x1e]
// CHECK: fcvt     h10, d11                   // encoding: [0x6a,0xc1,0x63,0x1e]
// CHECK: frintn   d12, d13                   // encoding: [0xac,0x41,0x64,0x1e]
// CHECK: frintp   d14, d15                   // encoding: [0xee,0xc1,0x64,0x1e]
// CHECK: frintm   d16, d17                   // encoding: [0x30,0x42,0x65,0x1e]
// CHECK: frintz   d18, d19                   // encoding: [0x72,0xc2,0x65,0x1e]
// CHECK: frinta   d20, d21                   // encoding: [0xb4,0x42,0x66,0x1e]
// CHECK: frintx   d22, d23                   // encoding: [0xf6,0x42,0x67,0x1e]
// CHECK: frinti   d24, d25                   // encoding: [0x38,0xc3,0x67,0x1e]

        fcvt s26, h27
        fcvt d28, h29
// CHECK: fcvt     s26, h27                   // encoding: [0x7a,0x43,0xe2,0x1e]
// CHECK: fcvt     d28, h29                   // encoding: [0xbc,0xc3,0xe2,0x1e]

//------------------------------------------------------------------------------
// Floating-point data-processing (2 sources)
//------------------------------------------------------------------------------

        fmul s20, s19, s17
        fdiv s1, s2, s3
        fadd s4, s5, s6
        fsub s7, s8, s9
        fmax s10, s11, s12
        fmin s13, s14, s15
        fmaxnm s16, s17, s18
        fminnm s19, s20, s21
        fnmul s22, s23, s24
// CHECK: fmul     s20, s19, s17              // encoding: [0x74,0x0a,0x31,0x1e]
// CHECK: fdiv     s1, s2, s3                 // encoding: [0x41,0x18,0x23,0x1e]
// CHECK: fadd     s4, s5, s6                 // encoding: [0xa4,0x28,0x26,0x1e]
// CHECK: fsub     s7, s8, s9                 // encoding: [0x07,0x39,0x29,0x1e]
// CHECK: fmax     s10, s11, s12              // encoding: [0x6a,0x49,0x2c,0x1e]
// CHECK: fmin     s13, s14, s15              // encoding: [0xcd,0x59,0x2f,0x1e]
// CHECK: fmaxnm   s16, s17, s18              // encoding: [0x30,0x6a,0x32,0x1e]
// CHECK: fminnm   s19, s20, s21              // encoding: [0x93,0x7a,0x35,0x1e]
// CHECK: fnmul    s22, s23, s24              // encoding: [0xf6,0x8a,0x38,0x1e]

        fmul d20, d19, d17
        fdiv d1, d2, d3
        fadd d4, d5, d6
        fsub d7, d8, d9
        fmax d10, d11, d12
        fmin d13, d14, d15
        fmaxnm d16, d17, d18
        fminnm d19, d20, d21
        fnmul d22, d23, d24
// CHECK: fmul     d20, d19, d17              // encoding: [0x74,0x0a,0x71,0x1e]
// CHECK: fdiv     d1, d2, d3                 // encoding: [0x41,0x18,0x63,0x1e]
// CHECK: fadd     d4, d5, d6                 // encoding: [0xa4,0x28,0x66,0x1e]
// CHECK: fsub     d7, d8, d9                 // encoding: [0x07,0x39,0x69,0x1e]
// CHECK: fmax     d10, d11, d12              // encoding: [0x6a,0x49,0x6c,0x1e]
// CHECK: fmin     d13, d14, d15              // encoding: [0xcd,0x59,0x6f,0x1e]
// CHECK: fmaxnm   d16, d17, d18              // encoding: [0x30,0x6a,0x72,0x1e]
// CHECK: fminnm   d19, d20, d21              // encoding: [0x93,0x7a,0x75,0x1e]
// CHECK: fnmul    d22, d23, d24              // encoding: [0xf6,0x8a,0x78,0x1e]

//------------------------------------------------------------------------------
// Floating-point data-processing (3 sources)
//------------------------------------------------------------------------------

        fmadd s3, s5, s6, s31
        fmadd d3, d13, d0, d23
        fmsub s3, s5, s6, s31
        fmsub d3, d13, d0, d23
        fnmadd s3, s5, s6, s31
        fnmadd d3, d13, d0, d23
        fnmsub s3, s5, s6, s31
        fnmsub d3, d13, d0, d23
// CHECK: fmadd   s3, s5, s6, s31         // encoding: [0xa3,0x7c,0x06,0x1f]
// CHECK: fmadd   d3, d13, d0, d23        // encoding: [0xa3,0x5d,0x40,0x1f]
// CHECK: fmsub   s3, s5, s6, s31         // encoding: [0xa3,0xfc,0x06,0x1f]
// CHECK: fmsub   d3, d13, d0, d23        // encoding: [0xa3,0xdd,0x40,0x1f]
// CHECK: fnmadd  s3, s5, s6, s31         // encoding: [0xa3,0x7c,0x26,0x1f]
// CHECK: fnmadd  d3, d13, d0, d23        // encoding: [0xa3,0x5d,0x60,0x1f]
// CHECK: fnmsub  s3, s5, s6, s31         // encoding: [0xa3,0xfc,0x26,0x1f]
// CHECK: fnmsub  d3, d13, d0, d23        // encoding: [0xa3,0xdd,0x60,0x1f]

//------------------------------------------------------------------------------
// Floating-point <-> fixed-point conversion
//------------------------------------------------------------------------------

        fcvtzs w3, s5, #1
        fcvtzs wzr, s20, #13
        fcvtzs w19, s0, #32
// CHECK: fcvtzs  w3, s5, #1              // encoding: [0xa3,0xfc,0x18,0x1e]
// CHECK: fcvtzs  wzr, s20, #13           // encoding: [0x9f,0xce,0x18,0x1e]
// CHECK: fcvtzs  w19, s0, #32            // encoding: [0x13,0x80,0x18,0x1e]

        fcvtzs x3, s5, #1
        fcvtzs x12, s30, #45
        fcvtzs x19, s0, #64
// CHECK: fcvtzs  x3, s5, #1              // encoding: [0xa3,0xfc,0x18,0x9e]
// CHECK: fcvtzs  x12, s30, #45           // encoding: [0xcc,0x4f,0x18,0x9e]
// CHECK: fcvtzs  x19, s0, #64            // encoding: [0x13,0x00,0x18,0x9e]

        fcvtzs w3, d5, #1
        fcvtzs wzr, d20, #13
        fcvtzs w19, d0, #32
// CHECK: fcvtzs  w3, d5, #1              // encoding: [0xa3,0xfc,0x58,0x1e]
// CHECK: fcvtzs  wzr, d20, #13           // encoding: [0x9f,0xce,0x58,0x1e]
// CHECK: fcvtzs  w19, d0, #32            // encoding: [0x13,0x80,0x58,0x1e]

        fcvtzs x3, d5, #1
        fcvtzs x12, d30, #45
        fcvtzs x19, d0, #64
// CHECK: fcvtzs  x3, d5, #1              // encoding: [0xa3,0xfc,0x58,0x9e]
// CHECK: fcvtzs  x12, d30, #45           // encoding: [0xcc,0x4f,0x58,0x9e]
// CHECK: fcvtzs  x19, d0, #64            // encoding: [0x13,0x00,0x58,0x9e]

        fcvtzu w3, s5, #1
        fcvtzu wzr, s20, #13
        fcvtzu w19, s0, #32
// CHECK: fcvtzu  w3, s5, #1              // encoding: [0xa3,0xfc,0x19,0x1e]
// CHECK: fcvtzu  wzr, s20, #13           // encoding: [0x9f,0xce,0x19,0x1e]
// CHECK: fcvtzu  w19, s0, #32            // encoding: [0x13,0x80,0x19,0x1e]

        fcvtzu x3, s5, #1
        fcvtzu x12, s30, #45
        fcvtzu x19, s0, #64
// CHECK: fcvtzu  x3, s5, #1              // encoding: [0xa3,0xfc,0x19,0x9e]
// CHECK: fcvtzu  x12, s30, #45           // encoding: [0xcc,0x4f,0x19,0x9e]
// CHECK: fcvtzu  x19, s0, #64            // encoding: [0x13,0x00,0x19,0x9e]

        fcvtzu w3, d5, #1
        fcvtzu wzr, d20, #13
        fcvtzu w19, d0, #32
// CHECK: fcvtzu  w3, d5, #1              // encoding: [0xa3,0xfc,0x59,0x1e]
// CHECK: fcvtzu  wzr, d20, #13           // encoding: [0x9f,0xce,0x59,0x1e]
// CHECK: fcvtzu  w19, d0, #32            // encoding: [0x13,0x80,0x59,0x1e]

        fcvtzu x3, d5, #1
        fcvtzu x12, d30, #45
        fcvtzu x19, d0, #64
// CHECK: fcvtzu  x3, d5, #1              // encoding: [0xa3,0xfc,0x59,0x9e]
// CHECK: fcvtzu  x12, d30, #45           // encoding: [0xcc,0x4f,0x59,0x9e]
// CHECK: fcvtzu  x19, d0, #64            // encoding: [0x13,0x00,0x59,0x9e]

        scvtf s23, w19, #1
        scvtf s31, wzr, #20
        scvtf s14, w0, #32
// CHECK: scvtf   s23, w19, #1            // encoding: [0x77,0xfe,0x02,0x1e]
// CHECK: scvtf   s31, wzr, #20           // encoding: [0xff,0xb3,0x02,0x1e]
// CHECK: scvtf   s14, w0, #32            // encoding: [0x0e,0x80,0x02,0x1e]

        scvtf s23, x19, #1
        scvtf s31, xzr, #20
        scvtf s14, x0, #64
// CHECK: scvtf   s23, x19, #1            // encoding: [0x77,0xfe,0x02,0x9e]
// CHECK: scvtf   s31, xzr, #20           // encoding: [0xff,0xb3,0x02,0x9e]
// CHECK: scvtf   s14, x0, #64            // encoding: [0x0e,0x00,0x02,0x9e]

        scvtf d23, w19, #1
        scvtf d31, wzr, #20
        scvtf d14, w0, #32
// CHECK: scvtf   d23, w19, #1            // encoding: [0x77,0xfe,0x42,0x1e]
// CHECK: scvtf   d31, wzr, #20           // encoding: [0xff,0xb3,0x42,0x1e]
// CHECK: scvtf   d14, w0, #32            // encoding: [0x0e,0x80,0x42,0x1e]

        scvtf d23, x19, #1
        scvtf d31, xzr, #20
        scvtf d14, x0, #64
// CHECK: scvtf   d23, x19, #1            // encoding: [0x77,0xfe,0x42,0x9e]
// CHECK: scvtf   d31, xzr, #20           // encoding: [0xff,0xb3,0x42,0x9e]
// CHECK: scvtf   d14, x0, #64            // encoding: [0x0e,0x00,0x42,0x9e]

        ucvtf s23, w19, #1
        ucvtf s31, wzr, #20
        ucvtf s14, w0, #32
// CHECK: ucvtf   s23, w19, #1            // encoding: [0x77,0xfe,0x03,0x1e]
// CHECK: ucvtf   s31, wzr, #20           // encoding: [0xff,0xb3,0x03,0x1e]
// CHECK: ucvtf   s14, w0, #32            // encoding: [0x0e,0x80,0x03,0x1e]

        ucvtf s23, x19, #1
        ucvtf s31, xzr, #20
        ucvtf s14, x0, #64
// CHECK: ucvtf   s23, x19, #1            // encoding: [0x77,0xfe,0x03,0x9e]
// CHECK: ucvtf   s31, xzr, #20           // encoding: [0xff,0xb3,0x03,0x9e]
// CHECK: ucvtf   s14, x0, #64            // encoding: [0x0e,0x00,0x03,0x9e]

        ucvtf d23, w19, #1
        ucvtf d31, wzr, #20
        ucvtf d14, w0, #32
// CHECK: ucvtf   d23, w19, #1            // encoding: [0x77,0xfe,0x43,0x1e]
// CHECK: ucvtf   d31, wzr, #20           // encoding: [0xff,0xb3,0x43,0x1e]
// CHECK: ucvtf   d14, w0, #32            // encoding: [0x0e,0x80,0x43,0x1e]

        ucvtf d23, x19, #1
        ucvtf d31, xzr, #20
        ucvtf d14, x0, #64
// CHECK: ucvtf   d23, x19, #1            // encoding: [0x77,0xfe,0x43,0x9e]
// CHECK: ucvtf   d31, xzr, #20           // encoding: [0xff,0xb3,0x43,0x9e]
// CHECK: ucvtf   d14, x0, #64            // encoding: [0x0e,0x00,0x43,0x9e]

//------------------------------------------------------------------------------
// Floating-point <-> integer conversion
//------------------------------------------------------------------------------
        fcvtns w3, s31
        fcvtns xzr, s12
        fcvtnu wzr, s12
        fcvtnu x0, s0
// CHECK: fcvtns   w3, s31                    // encoding: [0xe3,0x03,0x20,0x1e]
// CHECK: fcvtns   xzr, s12                   // encoding: [0x9f,0x01,0x20,0x9e]
// CHECK: fcvtnu   wzr, s12                   // encoding: [0x9f,0x01,0x21,0x1e]
// CHECK: fcvtnu   x0, s0                     // encoding: [0x00,0x00,0x21,0x9e]

        fcvtps wzr, s9
        fcvtps x12, s20
        fcvtpu w30, s23
        fcvtpu x29, s3
// CHECK: fcvtps   wzr, s9                    // encoding: [0x3f,0x01,0x28,0x1e]
// CHECK: fcvtps   x12, s20                   // encoding: [0x8c,0x02,0x28,0x9e]
// CHECK: fcvtpu   w30, s23                   // encoding: [0xfe,0x02,0x29,0x1e]
// CHECK: fcvtpu   x29, s3                    // encoding: [0x7d,0x00,0x29,0x9e]

        fcvtms w2, s3
        fcvtms x4, s5
        fcvtmu w6, s7
        fcvtmu x8, s9
// CHECK: fcvtms   w2, s3                     // encoding: [0x62,0x00,0x30,0x1e]
// CHECK: fcvtms   x4, s5                     // encoding: [0xa4,0x00,0x30,0x9e]
// CHECK: fcvtmu   w6, s7                     // encoding: [0xe6,0x00,0x31,0x1e]
// CHECK: fcvtmu   x8, s9                     // encoding: [0x28,0x01,0x31,0x9e]

        fcvtzs w10, s11
        fcvtzs x12, s13
        fcvtzu w14, s15
        fcvtzu x15, s16
// CHECK: fcvtzs   w10, s11                   // encoding: [0x6a,0x01,0x38,0x1e]
// CHECK: fcvtzs   x12, s13                   // encoding: [0xac,0x01,0x38,0x9e]
// CHECK: fcvtzu   w14, s15                   // encoding: [0xee,0x01,0x39,0x1e]
// CHECK: fcvtzu   x15, s16                   // encoding: [0x0f,0x02,0x39,0x9e]

        scvtf s17, w18
        scvtf s19, x20
        ucvtf s21, w22
        scvtf s23, x24
// CHECK: scvtf    s17, w18                   // encoding: [0x51,0x02,0x22,0x1e]
// CHECK: scvtf    s19, x20                   // encoding: [0x93,0x02,0x22,0x9e]
// CHECK: ucvtf    s21, w22                   // encoding: [0xd5,0x02,0x23,0x1e]
// CHECK: scvtf    s23, x24                   // encoding: [0x17,0x03,0x22,0x9e]

        fcvtas w25, s26
        fcvtas x27, s28
        fcvtau w29, s30
        fcvtau xzr, s0
// CHECK: fcvtas   w25, s26                   // encoding: [0x59,0x03,0x24,0x1e]
// CHECK: fcvtas   x27, s28                   // encoding: [0x9b,0x03,0x24,0x9e]
// CHECK: fcvtau   w29, s30                   // encoding: [0xdd,0x03,0x25,0x1e]
// CHECK: fcvtau   xzr, s0                    // encoding: [0x1f,0x00,0x25,0x9e]

        fcvtns w3, d31
        fcvtns xzr, d12
        fcvtnu wzr, d12
        fcvtnu x0, d0
// CHECK: fcvtns   w3, d31                    // encoding: [0xe3,0x03,0x60,0x1e]
// CHECK: fcvtns   xzr, d12                   // encoding: [0x9f,0x01,0x60,0x9e]
// CHECK: fcvtnu   wzr, d12                   // encoding: [0x9f,0x01,0x61,0x1e]
// CHECK: fcvtnu   x0, d0                     // encoding: [0x00,0x00,0x61,0x9e]

        fcvtps wzr, d9
        fcvtps x12, d20
        fcvtpu w30, d23
        fcvtpu x29, d3
// CHECK: fcvtps   wzr, d9                    // encoding: [0x3f,0x01,0x68,0x1e]
// CHECK: fcvtps   x12, d20                   // encoding: [0x8c,0x02,0x68,0x9e]
// CHECK: fcvtpu   w30, d23                   // encoding: [0xfe,0x02,0x69,0x1e]
// CHECK: fcvtpu   x29, d3                    // encoding: [0x7d,0x00,0x69,0x9e]

        fcvtms w2, d3
        fcvtms x4, d5
        fcvtmu w6, d7
        fcvtmu x8, d9
// CHECK: fcvtms   w2, d3                     // encoding: [0x62,0x00,0x70,0x1e]
// CHECK: fcvtms   x4, d5                     // encoding: [0xa4,0x00,0x70,0x9e]
// CHECK: fcvtmu   w6, d7                     // encoding: [0xe6,0x00,0x71,0x1e]
// CHECK: fcvtmu   x8, d9                     // encoding: [0x28,0x01,0x71,0x9e]

        fcvtzs w10, d11
        fcvtzs x12, d13
        fcvtzu w14, d15
        fcvtzu x15, d16
// CHECK: fcvtzs   w10, d11                   // encoding: [0x6a,0x01,0x78,0x1e]
// CHECK: fcvtzs   x12, d13                   // encoding: [0xac,0x01,0x78,0x9e]
// CHECK: fcvtzu   w14, d15                   // encoding: [0xee,0x01,0x79,0x1e]
// CHECK: fcvtzu   x15, d16                   // encoding: [0x0f,0x02,0x79,0x9e]

        scvtf d17, w18
        scvtf d19, x20
        ucvtf d21, w22
        ucvtf d23, x24
// CHECK: scvtf    d17, w18                   // encoding: [0x51,0x02,0x62,0x1e]
// CHECK: scvtf    d19, x20                   // encoding: [0x93,0x02,0x62,0x9e]
// CHECK: ucvtf    d21, w22                   // encoding: [0xd5,0x02,0x63,0x1e]
// CHECK: ucvtf    d23, x24                   // encoding: [0x17,0x03,0x63,0x9e]

        fcvtas w25, d26
        fcvtas x27, d28
        fcvtau w29, d30
        fcvtau xzr, d0
// CHECK: fcvtas   w25, d26                   // encoding: [0x59,0x03,0x64,0x1e]
// CHECK: fcvtas   x27, d28                   // encoding: [0x9b,0x03,0x64,0x9e]
// CHECK: fcvtau   w29, d30                   // encoding: [0xdd,0x03,0x65,0x1e]
// CHECK: fcvtau   xzr, d0                    // encoding: [0x1f,0x00,0x65,0x9e]

        fmov w3, s9
        fmov s9, w3
// CHECK: fmov     w3, s9                     // encoding: [0x23,0x01,0x26,0x1e]
// CHECK: fmov     s9, w3                     // encoding: [0x69,0x00,0x27,0x1e]

        fmov x20, d31
        fmov d1, x15
// CHECK: fmov     x20, d31                   // encoding: [0xf4,0x03,0x66,0x9e]
// CHECK: fmov     d1, x15                    // encoding: [0xe1,0x01,0x67,0x9e]

        fmov x3, v12.d[1]
        fmov v1.d[1], x19
        fmov v3.d[1], xzr
// CHECK: fmov     x3, v12.d[1]               // encoding: [0x83,0x01,0xae,0x9e]
// CHECK: fmov     v1.d[1], x19               // encoding: [0x61,0x02,0xaf,0x9e]
// CHECK: fmov     v3.d[1], xzr               // encoding: [0xe3,0x03,0xaf,0x9e]

//------------------------------------------------------------------------------
// Floating-point immediate
//------------------------------------------------------------------------------

        fmov s2, #0.125
        fmov s3, #1.0
        fmov d30, #16.0
// CHECK: fmov     s2, #{{0.12500000|1.250*e-01}}            // encoding: [0x02,0x10,0x28,0x1e]
// CHECK: fmov     s3, #{{1.00000000|1.0*e\+00}}            // encoding: [0x03,0x10,0x2e,0x1e]
// CHECK: fmov     d30, #{{16.00000000|1.60*e\+01}}          // encoding: [0x1e,0x10,0x66,0x1e]

        fmov s4, #1.0625
        fmov d10, #1.9375
// CHECK: fmov     s4, #{{1.06250*(e\+00)?}}            // encoding: [0x04,0x30,0x2e,0x1e]
// CHECK: fmov     d10, #{{1.93750*(e\+00)?}}           // encoding: [0x0a,0xf0,0x6f,0x1e]

        fmov s12, #-1.0
// CHECK: fmov     s12, #{{-1.0*(e\+00)?}}          // encoding: [0x0c,0x10,0x3e,0x1e]

        fmov d16, #8.5
// CHECK: fmov     d16, #{{8.50*(e\+00)?}}          // encoding: [0x10,0x30,0x64,0x1e]

//------------------------------------------------------------------------------
// Load-register (literal)
//------------------------------------------------------------------------------
        ldr w3, here
        ldr x29, there
        ldrsw xzr, everywhere

// CHECK: ldr    w3, here                // encoding: [0bAAA00011,A,A,0x18]
// CHECK:                                 //   fixup A - offset: 0, value: here, kind: fixup_aarch64_ldr_pcrel_imm19
// CHECK: ldr    x29, there              // encoding: [0bAAA11101,A,A,0x58]
// CHECK:                                 //   fixup A - offset: 0, value: there, kind: fixup_aarch64_ldr_pcrel_imm19
// CHECK: ldrsw    xzr, everywhere         // encoding: [0bAAA11111,A,A,0x98]
// CHECK:                                 //   fixup A - offset: 0, value: everywhere, kind: fixup_aarch64_ldr_pcrel_imm19

        ldr s0, who_knows
        ldr d0, i_dont
        ldr q0, there_must_be_a_better_way

// CHECK: ldr    s0, who_knows           // encoding: [0bAAA00000,A,A,0x1c]
// CHECK:                                 //   fixup A - offset: 0, value: who_knows, kind: fixup_aarch64_ldr_pcrel_imm19
// CHECK: ldr    d0, i_dont              // encoding: [0bAAA00000,A,A,0x5c]
// CHECK:                                 //   fixup A - offset: 0, value: i_dont, kind: fixup_aarch64_ldr_pcrel_imm19
// CHECK: ldr    q0, there_must_be_a_better_way // encoding: [0bAAA00000,A,A,0x9c]
// CHECK:                                 //   fixup A - offset: 0, value: there_must_be_a_better_way, kind: fixup_aarch64_ldr_pcrel_imm19

        ldr w0, #1048572
        ldr x10, #-1048576
// CHECK: ldr     w0, #1048572            // encoding: [0xe0,0xff,0x7f,0x18]
// CHECK: ldr     x10, #-1048576          // encoding: [0x0a,0x00,0x80,0x58]

        prfm pldl1strm, nowhere
        prfm #22, somewhere

// CHECK: prfm    pldl1strm, nowhere      // encoding: [0bAAA00001,A,A,0xd8]
// CHECK:                                 //   fixup A - offset: 0, value: nowhere, kind: fixup_aarch64_ldr_pcrel_imm19
// CHECK: prfm    #22, somewhere          // encoding: [0bAAA10110,A,A,0xd8]
// CHECK:                                 //   fixup A - offset: 0, value: somewhere, kind: fixup_aarch64_ldr_pcrel_imm19

//------------------------------------------------------------------------------
// Load/store exclusive
//------------------------------------------------------------------------------

        stxrb      w1, w2, [x3, #0]
        stxrh      w2, w3, [x4]
        stxr       wzr, w4, [sp]
        stxr       w5, x6, [x7]
// CHECK: stxrb    w1, w2, [x3]              // encoding: [0x62,0x7c,0x01,0x08]
// CHECK: stxrh    w2, w3, [x4]              // encoding: [0x83,0x7c,0x02,0x48]
// CHECK: stxr     wzr, w4, [sp]             // encoding: [0xe4,0x7f,0x1f,0x88]
// CHECK: stxr     w5, x6, [x7]              // encoding: [0xe6,0x7c,0x05,0xc8]

        ldxrb      w7, [x9]
        ldxrh      wzr, [x10]
        ldxr       w9, [sp]
        ldxr       x10, [x11]
// CHECK: ldxrb    w7, [x9]                  // encoding: [0x27,0x7d,0x5f,0x08]
// CHECK: ldxrh    wzr, [x10]                // encoding: [0x5f,0x7d,0x5f,0x48]
// CHECK: ldxr     w9, [sp]                  // encoding: [0xe9,0x7f,0x5f,0x88]
// CHECK: ldxr     x10, [x11]                // encoding: [0x6a,0x7d,0x5f,0xc8]

        stxp       w11, w12, w13, [x14]
        stxp       wzr, x23, x14, [x15]
// CHECK: stxp     w11, w12, w13, [x14]      // encoding: [0xcc,0x35,0x2b,0x88]
// CHECK: stxp     wzr, x23, x14, [x15]      // encoding: [0xf7,0x39,0x3f,0xc8]

        ldxp       w12, wzr, [sp]
        ldxp       x13, x14, [x15]
// CHECK: ldxp     w12, wzr, [sp]            // encoding: [0xec,0x7f,0x7f,0x88]
// CHECK: ldxp     x13, x14, [x15]           // encoding: [0xed,0x39,0x7f,0xc8]

        stlxrb     w14, w15, [x16]
        stlxrh     w15, w16, [x17,#0]
        stlxr      wzr, w17, [sp]
        stlxr      w18, x19, [x20]
// CHECK: stlxrb   w14, w15, [x16]           // encoding: [0x0f,0xfe,0x0e,0x08]
// CHECK: stlxrh   w15, w16, [x17]           // encoding: [0x30,0xfe,0x0f,0x48]
// CHECK: stlxr    wzr, w17, [sp]            // encoding: [0xf1,0xff,0x1f,0x88]
// CHECK: stlxr    w18, x19, [x20]           // encoding: [0x93,0xfe,0x12,0xc8]

        ldaxrb     w19, [x21]
        ldaxrh     w20, [sp]
        ldaxr      wzr, [x22]
        ldaxr      x21, [x23]
// CHECK: ldaxrb   w19, [x21]                // encoding: [0xb3,0xfe,0x5f,0x08]
// CHECK: ldaxrh   w20, [sp]                 // encoding: [0xf4,0xff,0x5f,0x48]
// CHECK: ldaxr    wzr, [x22]                // encoding: [0xdf,0xfe,0x5f,0x88]
// CHECK: ldaxr    x21, [x23]                // encoding: [0xf5,0xfe,0x5f,0xc8]

        stlxp      wzr, w22, w23, [x24]
        stlxp      w25, x26, x27, [sp]
// CHECK: stlxp    wzr, w22, w23, [x24]      // encoding: [0x16,0xdf,0x3f,0x88]
// CHECK: stlxp    w25, x26, x27, [sp]       // encoding: [0xfa,0xef,0x39,0xc8]

        ldaxp      w26, wzr, [sp]
        ldaxp      x27, x28, [x30]
// CHECK: ldaxp    w26, wzr, [sp]            // encoding: [0xfa,0xff,0x7f,0x88]
// CHECK: ldaxp    x27, x28, [x30]           // encoding: [0xdb,0xf3,0x7f,0xc8]

        stlrb      w27, [sp]
        stlrh      w28, [x0]
        stlr       wzr, [x1]
        stlr       x30, [x2]
// CHECK: stlrb    w27, [sp]                 // encoding: [0xfb,0xff,0x9f,0x08]
// CHECK: stlrh    w28, [x0]                 // encoding: [0x1c,0xfc,0x9f,0x48]
// CHECK: stlr     wzr, [x1]                 // encoding: [0x3f,0xfc,0x9f,0x88]
// CHECK: stlr     x30, [x2]                 // encoding: [0x5e,0xfc,0x9f,0xc8]

        ldarb      w29, [sp]
        ldarh      w30, [x0]
        ldar       wzr, [x1]
        ldar       x1, [x2]
// CHECK: ldarb    w29, [sp]                 // encoding: [0xfd,0xff,0xdf,0x08]
// CHECK: ldarh    w30, [x0]                 // encoding: [0x1e,0xfc,0xdf,0x48]
// CHECK: ldar     wzr, [x1]                 // encoding: [0x3f,0xfc,0xdf,0x88]
// CHECK: ldar     x1, [x2]                  // encoding: [0x41,0xfc,0xdf,0xc8]

        stlxp      wzr, w22, w23, [x24,#0]
// CHECK: stlxp    wzr, w22, w23, [x24]      // encoding: [0x16,0xdf,0x3f,0x88]

//------------------------------------------------------------------------------
// Load/store (unaligned immediate)
//------------------------------------------------------------------------------

        sturb w9, [sp, #0]
        sturh wzr, [x12, #255]
        stur w16, [x0, #-256]
        stur x28, [x14, #1]
// CHECK: sturb    w9, [sp]                   // encoding: [0xe9,0x03,0x00,0x38]
// CHECK: sturh    wzr, [x12, #255]           // encoding: [0x9f,0xf1,0x0f,0x78]
// CHECK: stur     w16, [x0, #-256]           // encoding: [0x10,0x00,0x10,0xb8]
// CHECK: stur     x28, [x14, #1]             // encoding: [0xdc,0x11,0x00,0xf8]

        ldurb w1, [x20, #255]
        ldurh w20, [x1, #255]
        ldur w12, [sp, #255]
        ldur xzr, [x12, #255]
// CHECK: ldurb    w1, [x20, #255]            // encoding: [0x81,0xf2,0x4f,0x38]
// CHECK: ldurh    w20, [x1, #255]            // encoding: [0x34,0xf0,0x4f,0x78]
// CHECK: ldur     w12, [sp, #255]            // encoding: [0xec,0xf3,0x4f,0xb8]
// CHECK: ldur     xzr, [x12, #255]           // encoding: [0x9f,0xf1,0x4f,0xf8]

        ldursb x9, [x7, #-256]
        ldursh x17, [x19, #-256]
        ldursw x20, [x15, #-256]
        ldursw x13, [x2]
        prfum pldl2keep, [sp, #-256]
        ldursb w19, [x1, #-256]
        ldursh w15, [x21, #-256]
// CHECK: ldursb   x9, [x7, #-256]            // encoding: [0xe9,0x00,0x90,0x38]
// CHECK: ldursh   x17, [x19, #-256]          // encoding: [0x71,0x02,0x90,0x78]
// CHECK: ldursw   x20, [x15, #-256]          // encoding: [0xf4,0x01,0x90,0xb8]
// CHECK: ldursw   x13, [x2]                  // encoding: [0x4d,0x00,0x80,0xb8]
// CHECK: prfum    pldl2keep, [sp, #-256]     // encoding: [0xe2,0x03,0x90,0xf8]
// CHECK: ldursb   w19, [x1, #-256]           // encoding: [0x33,0x00,0xd0,0x38]
// CHECK: ldursh   w15, [x21, #-256]          // encoding: [0xaf,0x02,0xd0,0x78]

        stur b0, [sp, #1]
        stur h12, [x12, #-1]
        stur s15, [x0, #255]
        stur d31, [x5, #25]
        stur q9, [x5]
// CHECK: stur     b0, [sp, #1]               // encoding: [0xe0,0x13,0x00,0x3c]
// CHECK: stur     h12, [x12, #-1]            // encoding: [0x8c,0xf1,0x1f,0x7c]
// CHECK: stur     s15, [x0, #255]            // encoding: [0x0f,0xf0,0x0f,0xbc]
// CHECK: stur     d31, [x5, #25]             // encoding: [0xbf,0x90,0x01,0xfc]
// CHECK: stur     q9, [x5]                   // encoding: [0xa9,0x00,0x80,0x3c]

        ldur b3, [sp]
        ldur h5, [x4, #-256]
        ldur s7, [x12, #-1]
        ldur d11, [x19, #4]
        ldur q13, [x1, #2]
// CHECK: ldur     b3, [sp]                   // encoding: [0xe3,0x03,0x40,0x3c]
// CHECK: ldur     h5, [x4, #-256]            // encoding: [0x85,0x00,0x50,0x7c]
// CHECK: ldur     s7, [x12, #-1]             // encoding: [0x87,0xf1,0x5f,0xbc]
// CHECK: ldur     d11, [x19, #4]             // encoding: [0x6b,0x42,0x40,0xfc]
// CHECK: ldur     q13, [x1, #2]              // encoding: [0x2d,0x20,0xc0,0x3c]

//------------------------------------------------------------------------------
// Load/store (unsigned immediate)
//------------------------------------------------------------------------------

//// Basic addressing mode limits: 8 byte access
        ldr x0, [x0]
        ldr x4, [x29, #0]
        ldr x30, [x12, #32760]
        ldr x20, [sp, #8]
// CHECK: ldr      x0, [x0]                   // encoding: [0x00,0x00,0x40,0xf9]
// CHECK: ldr      x4, [x29]                  // encoding: [0xa4,0x03,0x40,0xf9]
// CHECK: ldr      x30, [x12, #32760]         // encoding: [0x9e,0xfd,0x7f,0xf9]
// CHECK: ldr      x20, [sp, #8]              // encoding: [0xf4,0x07,0x40,0xf9]

//// Rt treats 31 as zero-register
        ldr xzr, [sp]
// CHECK: ldr      xzr, [sp]                  // encoding: [0xff,0x03,0x40,0xf9]

        //// 4-byte load, check still 64-bit address, limits
        ldr w2, [sp]
        ldr w17, [sp, #16380]
        ldr w13, [x2, #4]
// CHECK: ldr      w2, [sp]                   // encoding: [0xe2,0x03,0x40,0xb9]
// CHECK: ldr      w17, [sp, #16380]          // encoding: [0xf1,0xff,0x7f,0xb9]
// CHECK: ldr      w13, [x2, #4]              // encoding: [0x4d,0x04,0x40,0xb9]

//// Signed 4-byte load. Limits.
        ldrsw x2, [x5,#4]
        ldrsw x23, [sp, #16380]
// CHECK: ldrsw    x2, [x5, #4]               // encoding: [0xa2,0x04,0x80,0xb9]
// CHECK: ldrsw    x23, [sp, #16380]          // encoding: [0xf7,0xff,0xbf,0xb9]

////  2-byte loads
        ldrh w2, [x4]
        ldrsh w23, [x6, #8190]
        ldrsh wzr, [sp, #2]
        ldrsh x29, [x2, #2]
// CHECK: ldrh     w2, [x4]                   // encoding: [0x82,0x00,0x40,0x79]
// CHECK: ldrsh    w23, [x6, #8190]           // encoding: [0xd7,0xfc,0xff,0x79]
// CHECK: ldrsh    wzr, [sp, #2]              // encoding: [0xff,0x07,0xc0,0x79]
// CHECK: ldrsh    x29, [x2, #2]              // encoding: [0x5d,0x04,0x80,0x79]

//// 1-byte loads
        ldrb w26, [x3, #121]
        ldrb w12, [x2, #0]
        ldrsb w27, [sp, #4095]
        ldrsb xzr, [x15]
// CHECK: ldrb     w26, [x3, #121]            // encoding: [0x7a,0xe4,0x41,0x39]
// CHECK: ldrb     w12, [x2]                  // encoding: [0x4c,0x00,0x40,0x39]
// CHECK: ldrsb    w27, [sp, #4095]           // encoding: [0xfb,0xff,0xff,0x39]
// CHECK: ldrsb    xzr, [x15]                 // encoding: [0xff,0x01,0x80,0x39]

//// Stores
        str x30, [sp]
        str w20, [x4, #16380]
        strh w20, [x10, #14]
        strh w17, [sp, #8190]
        strb w23, [x3, #4095]
        strb wzr, [x2]
// CHECK: str      x30, [sp]                  // encoding: [0xfe,0x03,0x00,0xf9]
// CHECK: str      w20, [x4, #16380]          // encoding: [0x94,0xfc,0x3f,0xb9]
// CHECK: strh     w20, [x10, #14]            // encoding: [0x54,0x1d,0x00,0x79]
// CHECK: strh     w17, [sp, #8190]           // encoding: [0xf1,0xff,0x3f,0x79]
// CHECK: strb     w23, [x3, #4095]           // encoding: [0x77,0xfc,0x3f,0x39]
// CHECK: strb     wzr, [x2]                  // encoding: [0x5f,0x00,0x00,0x39]

//// Relocations
        str x15, [x5, #:lo12:sym]
        ldrb w15, [x5, #:lo12:sym]
        ldrsh x15, [x5, #:lo12:sym]
        ldrsw x15, [x5, #:lo12:sym]
        ldr x15, [x5, #:lo12:sym]
        ldr q3, [x2, #:lo12:sym]

// CHECK: str    x15, [x5, :lo12:sym]    // encoding: [0xaf,0bAAAAAA00,0b00AAAAAA,0xf9]
// CHECK:                                 //   fixup A - offset: 0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK: ldrb    w15, [x5, :lo12:sym]    // encoding: [0xaf,0bAAAAAA00,0b01AAAAAA,0x39]
// CHECK:                                 //   fixup A - offset: 0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale1
// CHECK: ldrsh    x15, [x5, :lo12:sym]    // encoding: [0xaf,0bAAAAAA00,0b10AAAAAA,0x79]
// CHECK:                                 //   fixup A - offset: 0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale2
// CHECK: ldrsw    x15, [x5, :lo12:sym]    // encoding: [0xaf,0bAAAAAA00,0b10AAAAAA,0xb9]
// CHECK:                                 //   fixup A - offset: 0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK: ldr    x15, [x5, :lo12:sym]    // encoding: [0xaf,0bAAAAAA00,0b01AAAAAA,0xf9]
// CHECK:                                 //   fixup A - offset: 0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK: ldr    q3, [x2, :lo12:sym]     // encoding: [0x43,0bAAAAAA00,0b11AAAAAA,0x3d]
// CHECK:                                 //   fixup A - offset: 0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale16

        prfm pldl1keep, [sp, #8]
        prfm pldl1strm, [x3]
        prfm pldl2keep, [x5,#16]
        prfm pldl2strm, [x2]
        prfm pldl3keep, [x5]
        prfm pldl3strm, [x6]
        prfm plil1keep, [sp, #8]
        prfm plil1strm, [x3]
        prfm plil2keep, [x5,#16]
        prfm plil2strm, [x2]
        prfm plil3keep, [x5]
        prfm plil3strm, [x6]
        prfm pstl1keep, [sp, #8]
        prfm pstl1strm, [x3]
        prfm pstl2keep, [x5,#16]
        prfm pstl2strm, [x2]
        prfm pstl3keep, [x5]
        prfm pstl3strm, [x6]
        prfm #15, [sp]
// CHECK: prfm    pldl1keep, [sp, #8]     // encoding: [0xe0,0x07,0x80,0xf9]
// CHECK: prfm    pldl1strm, [x3{{(, #0)?}}]     // encoding: [0x61,0x00,0x80,0xf9]
// CHECK: prfm    pldl2keep, [x5, #16]    // encoding: [0xa2,0x08,0x80,0xf9]
// CHECK: prfm    pldl2strm, [x2{{(, #0)?}}]     // encoding: [0x43,0x00,0x80,0xf9]
// CHECK: prfm    pldl3keep, [x5{{(, #0)?}}]     // encoding: [0xa4,0x00,0x80,0xf9]
// CHECK: prfm    pldl3strm, [x6{{(, #0)?}}]     // encoding: [0xc5,0x00,0x80,0xf9]
// CHECK: prfm    plil1keep, [sp, #8]     // encoding: [0xe8,0x07,0x80,0xf9]
// CHECK: prfm    plil1strm, [x3{{(, #0)?}}]     // encoding: [0x69,0x00,0x80,0xf9]
// CHECK: prfm    plil2keep, [x5, #16]    // encoding: [0xaa,0x08,0x80,0xf9]
// CHECK: prfm    plil2strm, [x2{{(, #0)?}}]     // encoding: [0x4b,0x00,0x80,0xf9]
// CHECK: prfm    plil3keep, [x5{{(, #0)?}}]     // encoding: [0xac,0x00,0x80,0xf9]
// CHECK: prfm    plil3strm, [x6{{(, #0)?}}]     // encoding: [0xcd,0x00,0x80,0xf9]
// CHECK: prfm    pstl1keep, [sp, #8]     // encoding: [0xf0,0x07,0x80,0xf9]
// CHECK: prfm    pstl1strm, [x3{{(, #0)?}}]     // encoding: [0x71,0x00,0x80,0xf9]
// CHECK: prfm    pstl2keep, [x5, #16]    // encoding: [0xb2,0x08,0x80,0xf9]
// CHECK: prfm    pstl2strm, [x2{{(, #0)?}}]     // encoding: [0x53,0x00,0x80,0xf9]
// CHECK: prfm    pstl3keep, [x5{{(, #0)?}}]     // encoding: [0xb4,0x00,0x80,0xf9]
// CHECK: prfm    pstl3strm, [x6{{(, #0)?}}]     // encoding: [0xd5,0x00,0x80,0xf9]
// CHECK: prfm    #15, [sp{{(, #0)?}}]           // encoding: [0xef,0x03,0x80,0xf9]

//// Floating-point versions

        ldr b31, [sp, #4095]
        ldr h20, [x2, #8190]
        ldr s10, [x19, #16380]
        ldr d3, [x10, #32760]
        str q12, [sp, #65520]
// CHECK: ldr      b31, [sp, #4095]           // encoding: [0xff,0xff,0x7f,0x3d]
// CHECK: ldr      h20, [x2, #8190]           // encoding: [0x54,0xfc,0x7f,0x7d]
// CHECK: ldr      s10, [x19, #16380]         // encoding: [0x6a,0xfe,0x7f,0xbd]
// CHECK: ldr      d3, [x10, #32760]          // encoding: [0x43,0xfd,0x7f,0xfd]
// CHECK: str      q12, [sp, #65520]          // encoding: [0xec,0xff,0xbf,0x3d]

//------------------------------------------------------------------------------
// Load/store register (register offset)
//------------------------------------------------------------------------------

        ldrb w3, [sp, x5]
        ldrb w9, [x27, x6, lsl #0]
        ldrsb w10, [x30, x7]
        ldrb w11, [x29, x3, sxtx]
        strb w12, [x28, xzr, sxtx #0]
        ldrb w14, [x26, w6, uxtw]
        ldrsb w15, [x25, w7, uxtw #0]
        ldrb w17, [x23, w9, sxtw]
        ldrsb x18, [x22, w10, sxtw #0]
// CHECK: ldrb     w3, [sp, x5]               // encoding: [0xe3,0x6b,0x65,0x38]
// CHECK: ldrb     w9, [x27, x6, lsl #0]      // encoding: [0x69,0x7b,0x66,0x38]
// CHECK: ldrsb    w10, [x30, x7]             // encoding: [0xca,0x6b,0xe7,0x38]
// CHECK: ldrb     w11, [x29, x3, sxtx]       // encoding: [0xab,0xeb,0x63,0x38]
// CHECK: strb     w12, [x28, xzr, sxtx #0]   // encoding: [0x8c,0xfb,0x3f,0x38]
// CHECK: ldrb     w14, [x26, w6, uxtw]       // encoding: [0x4e,0x4b,0x66,0x38]
// CHECK: ldrsb    w15, [x25, w7, uxtw #0]    // encoding: [0x2f,0x5b,0xe7,0x38]
// CHECK: ldrb     w17, [x23, w9, sxtw]       // encoding: [0xf1,0xca,0x69,0x38]
// CHECK: ldrsb    x18, [x22, w10, sxtw #0]   // encoding: [0xd2,0xda,0xaa,0x38]

        ldrsh w3, [sp, x5]
        ldrsh w9, [x27, x6, lsl #0]
        ldrh w10, [x30, x7, lsl #1]
        strh w11, [x29, x3, sxtx]
        ldrh w12, [x28, xzr, sxtx #0]
        ldrsh x13, [x27, x5, sxtx #1]
        ldrh w14, [x26, w6, uxtw]
        ldrh w15, [x25, w7, uxtw #0]
        ldrsh w16, [x24, w8, uxtw #1]
        ldrh w17, [x23, w9, sxtw]
        ldrh w18, [x22, w10, sxtw #0]
        strh w19, [x21, wzr, sxtw #1]
// CHECK: ldrsh    w3, [sp, x5]               // encoding: [0xe3,0x6b,0xe5,0x78]
// CHECK: ldrsh    w9, [x27, x6]              // encoding: [0x69,0x6b,0xe6,0x78]
// CHECK: ldrh     w10, [x30, x7, lsl #1]     // encoding: [0xca,0x7b,0x67,0x78]
// CHECK: strh     w11, [x29, x3, sxtx]       // encoding: [0xab,0xeb,0x23,0x78]
// CHECK: ldrh     w12, [x28, xzr, sxtx]      // encoding: [0x8c,0xeb,0x7f,0x78]
// CHECK: ldrsh    x13, [x27, x5, sxtx #1]    // encoding: [0x6d,0xfb,0xa5,0x78]
// CHECK: ldrh     w14, [x26, w6, uxtw]       // encoding: [0x4e,0x4b,0x66,0x78]
// CHECK: ldrh     w15, [x25, w7, uxtw]       // encoding: [0x2f,0x4b,0x67,0x78]
// CHECK: ldrsh    w16, [x24, w8, uxtw #1]    // encoding: [0x10,0x5b,0xe8,0x78]
// CHECK: ldrh     w17, [x23, w9, sxtw]       // encoding: [0xf1,0xca,0x69,0x78]
// CHECK: ldrh     w18, [x22, w10, sxtw]      // encoding: [0xd2,0xca,0x6a,0x78]
// CHECK: strh     w19, [x21, wzr, sxtw #1]   // encoding: [0xb3,0xda,0x3f,0x78]

        ldr w3, [sp, x5]
        ldr s9, [x27, x6, lsl #0]
        ldr w10, [x30, x7, lsl #2]
        ldr w11, [x29, x3, sxtx]
        str s12, [x28, xzr, sxtx #0]
        str w13, [x27, x5, sxtx #2]
        str w14, [x26, w6, uxtw]
        ldr w15, [x25, w7, uxtw #0]
        ldr w16, [x24, w8, uxtw #2]
        ldrsw x17, [x23, w9, sxtw]
        ldr w18, [x22, w10, sxtw #0]
        ldrsw x19, [x21, wzr, sxtw #2]
// CHECK: ldr      w3, [sp, x5]               // encoding: [0xe3,0x6b,0x65,0xb8]
// CHECK: ldr      s9, [x27, x6]              // encoding: [0x69,0x6b,0x66,0xbc]
// CHECK: ldr      w10, [x30, x7, lsl #2]     // encoding: [0xca,0x7b,0x67,0xb8]
// CHECK: ldr      w11, [x29, x3, sxtx]       // encoding: [0xab,0xeb,0x63,0xb8]
// CHECK: str      s12, [x28, xzr, sxtx]      // encoding: [0x8c,0xeb,0x3f,0xbc]
// CHECK: str      w13, [x27, x5, sxtx #2]    // encoding: [0x6d,0xfb,0x25,0xb8]
// CHECK: str      w14, [x26, w6, uxtw]       // encoding: [0x4e,0x4b,0x26,0xb8]
// CHECK: ldr      w15, [x25, w7, uxtw]       // encoding: [0x2f,0x4b,0x67,0xb8]
// CHECK: ldr      w16, [x24, w8, uxtw #2]    // encoding: [0x10,0x5b,0x68,0xb8]
// CHECK: ldrsw    x17, [x23, w9, sxtw]       // encoding: [0xf1,0xca,0xa9,0xb8]
// CHECK: ldr      w18, [x22, w10, sxtw]      // encoding: [0xd2,0xca,0x6a,0xb8]
// CHECK: ldrsw    x19, [x21, wzr, sxtw #2]   // encoding: [0xb3,0xda,0xbf,0xb8]

        ldr x3, [sp, x5]
        str x9, [x27, x6, lsl #0]
        ldr d10, [x30, x7, lsl #3]
        str x11, [x29, x3, sxtx]
        ldr x12, [x28, xzr, sxtx #0]
        ldr x13, [x27, x5, sxtx #3]
        prfm pldl1keep, [x26, w6, uxtw]
        ldr x15, [x25, w7, uxtw #0]
        ldr x16, [x24, w8, uxtw #3]
        ldr x17, [x23, w9, sxtw]
        ldr x18, [x22, w10, sxtw #0]
        str d19, [x21, wzr, sxtw #3]
        prfm #6, [x0, x5]
// CHECK: ldr      x3, [sp, x5]               // encoding: [0xe3,0x6b,0x65,0xf8]
// CHECK: str      x9, [x27, x6]              // encoding: [0x69,0x6b,0x26,0xf8]
// CHECK: ldr      d10, [x30, x7, lsl #3]     // encoding: [0xca,0x7b,0x67,0xfc]
// CHECK: str      x11, [x29, x3, sxtx]       // encoding: [0xab,0xeb,0x23,0xf8]
// CHECK: ldr      x12, [x28, xzr, sxtx]      // encoding: [0x8c,0xeb,0x7f,0xf8]
// CHECK: ldr      x13, [x27, x5, sxtx #3]    // encoding: [0x6d,0xfb,0x65,0xf8]
// CHECK: prfm     pldl1keep, [x26, w6, uxtw] // encoding: [0x40,0x4b,0xa6,0xf8]
// CHECK: ldr      x15, [x25, w7, uxtw]       // encoding: [0x2f,0x4b,0x67,0xf8]
// CHECK: ldr      x16, [x24, w8, uxtw #3]    // encoding: [0x10,0x5b,0x68,0xf8]
// CHECK: ldr      x17, [x23, w9, sxtw]       // encoding: [0xf1,0xca,0x69,0xf8]
// CHECK: ldr      x18, [x22, w10, sxtw]      // encoding: [0xd2,0xca,0x6a,0xf8]
// CHECK: str      d19, [x21, wzr, sxtw #3]   // encoding: [0xb3,0xda,0x3f,0xfc]
// CHECK: prfm     #6, [x0, x5{{(, lsl #0)?}}]       // encoding: [0x06,0x68,0xa5,0xf8]

        ldr q3, [sp, x5]
        ldr q9, [x27, x6, lsl #0]
        ldr q10, [x30, x7, lsl #4]
        str q11, [x29, x3, sxtx]
        str q12, [x28, xzr, sxtx #0]
        str q13, [x27, x5, sxtx #4]
        ldr q14, [x26, w6, uxtw]
        ldr q15, [x25, w7, uxtw #0]
        ldr q16, [x24, w8, uxtw #4]
        ldr q17, [x23, w9, sxtw]
        str q18, [x22, w10, sxtw #0]
        ldr q19, [x21, wzr, sxtw #4]
// CHECK: ldr      q3, [sp, x5]               // encoding: [0xe3,0x6b,0xe5,0x3c]
// CHECK: ldr      q9, [x27, x6]              // encoding: [0x69,0x6b,0xe6,0x3c]
// CHECK: ldr      q10, [x30, x7, lsl #4]     // encoding: [0xca,0x7b,0xe7,0x3c]
// CHECK: str      q11, [x29, x3, sxtx]       // encoding: [0xab,0xeb,0xa3,0x3c]
// CHECK: str      q12, [x28, xzr, sxtx]      // encoding: [0x8c,0xeb,0xbf,0x3c]
// CHECK: str      q13, [x27, x5, sxtx #4]    // encoding: [0x6d,0xfb,0xa5,0x3c]
// CHECK: ldr      q14, [x26, w6, uxtw]       // encoding: [0x4e,0x4b,0xe6,0x3c]
// CHECK: ldr      q15, [x25, w7, uxtw]       // encoding: [0x2f,0x4b,0xe7,0x3c]
// CHECK: ldr      q16, [x24, w8, uxtw #4]    // encoding: [0x10,0x5b,0xe8,0x3c]
// CHECK: ldr      q17, [x23, w9, sxtw]       // encoding: [0xf1,0xca,0xe9,0x3c]
// CHECK: str      q18, [x22, w10, sxtw]      // encoding: [0xd2,0xca,0xaa,0x3c]
// CHECK: ldr      q19, [x21, wzr, sxtw #4]   // encoding: [0xb3,0xda,0xff,0x3c]

//------------------------------------------------------------------------------
// Load/store register (immediate post-indexed)
//------------------------------------------------------------------------------

        strb w9, [x2], #255
        strb w10, [x3], #1
        strb w10, [x3], #-256
        strh w9, [x2], #255
        strh w9, [x2], #1
        strh w10, [x3], #-256
// CHECK: strb     w9, [x2], #255             // encoding: [0x49,0xf4,0x0f,0x38]
// CHECK: strb     w10, [x3], #1              // encoding: [0x6a,0x14,0x00,0x38]
// CHECK: strb     w10, [x3], #-256           // encoding: [0x6a,0x04,0x10,0x38]
// CHECK: strh     w9, [x2], #255             // encoding: [0x49,0xf4,0x0f,0x78]
// CHECK: strh     w9, [x2], #1               // encoding: [0x49,0x14,0x00,0x78]
// CHECK: strh     w10, [x3], #-256           // encoding: [0x6a,0x04,0x10,0x78]

        str w19, [sp], #255
        str w20, [x30], #1
        str w21, [x12], #-256
        str xzr, [x9], #255
        str x2, [x3], #1
        str x19, [x12], #-256
// CHECK: str      w19, [sp], #255            // encoding: [0xf3,0xf7,0x0f,0xb8]
// CHECK: str      w20, [x30], #1             // encoding: [0xd4,0x17,0x00,0xb8]
// CHECK: str      w21, [x12], #-256          // encoding: [0x95,0x05,0x10,0xb8]
// CHECK: str      xzr, [x9], #255            // encoding: [0x3f,0xf5,0x0f,0xf8]
// CHECK: str      x2, [x3], #1               // encoding: [0x62,0x14,0x00,0xf8]
// CHECK: str      x19, [x12], #-256          // encoding: [0x93,0x05,0x10,0xf8]

        ldrb w9, [x2], #255
        ldrb w10, [x3], #1
        ldrb w10, [x3], #-256
        ldrh w9, [x2], #255
        ldrh w9, [x2], #1
        ldrh w10, [x3], #-256
// CHECK: ldrb     w9, [x2], #255             // encoding: [0x49,0xf4,0x4f,0x38]
// CHECK: ldrb     w10, [x3], #1              // encoding: [0x6a,0x14,0x40,0x38]
// CHECK: ldrb     w10, [x3], #-256           // encoding: [0x6a,0x04,0x50,0x38]
// CHECK: ldrh     w9, [x2], #255             // encoding: [0x49,0xf4,0x4f,0x78]
// CHECK: ldrh     w9, [x2], #1               // encoding: [0x49,0x14,0x40,0x78]
// CHECK: ldrh     w10, [x3], #-256           // encoding: [0x6a,0x04,0x50,0x78]

        ldr w19, [sp], #255
        ldr w20, [x30], #1
        ldr w21, [x12], #-256
        ldr xzr, [x9], #255
        ldr x2, [x3], #1
        ldr x19, [x12], #-256
// CHECK: ldr      w19, [sp], #255            // encoding: [0xf3,0xf7,0x4f,0xb8]
// CHECK: ldr      w20, [x30], #1             // encoding: [0xd4,0x17,0x40,0xb8]
// CHECK: ldr      w21, [x12], #-256          // encoding: [0x95,0x05,0x50,0xb8]
// CHECK: ldr      xzr, [x9], #255            // encoding: [0x3f,0xf5,0x4f,0xf8]
// CHECK: ldr      x2, [x3], #1               // encoding: [0x62,0x14,0x40,0xf8]
// CHECK: ldr      x19, [x12], #-256          // encoding: [0x93,0x05,0x50,0xf8]

        ldrsb xzr, [x9], #255
        ldrsb x2, [x3], #1
        ldrsb x19, [x12], #-256
        ldrsh xzr, [x9], #255
        ldrsh x2, [x3], #1
        ldrsh x19, [x12], #-256
        ldrsw xzr, [x9], #255
        ldrsw x2, [x3], #1
        ldrsw x19, [x12], #-256
// CHECK: ldrsb    xzr, [x9], #255            // encoding: [0x3f,0xf5,0x8f,0x38]
// CHECK: ldrsb    x2, [x3], #1               // encoding: [0x62,0x14,0x80,0x38]
// CHECK: ldrsb    x19, [x12], #-256          // encoding: [0x93,0x05,0x90,0x38]
// CHECK: ldrsh    xzr, [x9], #255            // encoding: [0x3f,0xf5,0x8f,0x78]
// CHECK: ldrsh    x2, [x3], #1               // encoding: [0x62,0x14,0x80,0x78]
// CHECK: ldrsh    x19, [x12], #-256          // encoding: [0x93,0x05,0x90,0x78]
// CHECK: ldrsw    xzr, [x9], #255            // encoding: [0x3f,0xf5,0x8f,0xb8]
// CHECK: ldrsw    x2, [x3], #1               // encoding: [0x62,0x14,0x80,0xb8]
// CHECK: ldrsw    x19, [x12], #-256          // encoding: [0x93,0x05,0x90,0xb8]

        ldrsb wzr, [x9], #255
        ldrsb w2, [x3], #1
        ldrsb w19, [x12], #-256
        ldrsh wzr, [x9], #255
        ldrsh w2, [x3], #1
        ldrsh w19, [x12], #-256
// CHECK: ldrsb    wzr, [x9], #255            // encoding: [0x3f,0xf5,0xcf,0x38]
// CHECK: ldrsb    w2, [x3], #1               // encoding: [0x62,0x14,0xc0,0x38]
// CHECK: ldrsb    w19, [x12], #-256          // encoding: [0x93,0x05,0xd0,0x38]
// CHECK: ldrsh    wzr, [x9], #255            // encoding: [0x3f,0xf5,0xcf,0x78]
// CHECK: ldrsh    w2, [x3], #1               // encoding: [0x62,0x14,0xc0,0x78]
// CHECK: ldrsh    w19, [x12], #-256          // encoding: [0x93,0x05,0xd0,0x78]

        str b0, [x0], #255
        str b3, [x3], #1
        str b5, [sp], #-256
        str h10, [x10], #255
        str h13, [x23], #1
        str h15, [sp], #-256
        str s20, [x20], #255
        str s23, [x23], #1
        str s25, [x0], #-256
        str d20, [x20], #255
        str d23, [x23], #1
        str d25, [x0], #-256
// CHECK: str      b0, [x0], #255             // encoding: [0x00,0xf4,0x0f,0x3c]
// CHECK: str      b3, [x3], #1               // encoding: [0x63,0x14,0x00,0x3c]
// CHECK: str      b5, [sp], #-256            // encoding: [0xe5,0x07,0x10,0x3c]
// CHECK: str      h10, [x10], #255           // encoding: [0x4a,0xf5,0x0f,0x7c]
// CHECK: str      h13, [x23], #1             // encoding: [0xed,0x16,0x00,0x7c]
// CHECK: str      h15, [sp], #-256           // encoding: [0xef,0x07,0x10,0x7c]
// CHECK: str      s20, [x20], #255           // encoding: [0x94,0xf6,0x0f,0xbc]
// CHECK: str      s23, [x23], #1             // encoding: [0xf7,0x16,0x00,0xbc]
// CHECK: str      s25, [x0], #-256           // encoding: [0x19,0x04,0x10,0xbc]
// CHECK: str      d20, [x20], #255           // encoding: [0x94,0xf6,0x0f,0xfc]
// CHECK: str      d23, [x23], #1             // encoding: [0xf7,0x16,0x00,0xfc]
// CHECK: str      d25, [x0], #-256           // encoding: [0x19,0x04,0x10,0xfc]

        ldr b0, [x0], #255
        ldr b3, [x3], #1
        ldr b5, [sp], #-256
        ldr h10, [x10], #255
        ldr h13, [x23], #1
        ldr h15, [sp], #-256
        ldr s20, [x20], #255
        ldr s23, [x23], #1
        ldr s25, [x0], #-256
        ldr d20, [x20], #255
        ldr d23, [x23], #1
        ldr d25, [x0], #-256
// CHECK: ldr      b0, [x0], #255             // encoding: [0x00,0xf4,0x4f,0x3c]
// CHECK: ldr      b3, [x3], #1               // encoding: [0x63,0x14,0x40,0x3c]
// CHECK: ldr      b5, [sp], #-256            // encoding: [0xe5,0x07,0x50,0x3c]
// CHECK: ldr      h10, [x10], #255           // encoding: [0x4a,0xf5,0x4f,0x7c]
// CHECK: ldr      h13, [x23], #1             // encoding: [0xed,0x16,0x40,0x7c]
// CHECK: ldr      h15, [sp], #-256           // encoding: [0xef,0x07,0x50,0x7c]
// CHECK: ldr      s20, [x20], #255           // encoding: [0x94,0xf6,0x4f,0xbc]
// CHECK: ldr      s23, [x23], #1             // encoding: [0xf7,0x16,0x40,0xbc]
// CHECK: ldr      s25, [x0], #-256           // encoding: [0x19,0x04,0x50,0xbc]
// CHECK: ldr      d20, [x20], #255           // encoding: [0x94,0xf6,0x4f,0xfc]
// CHECK: ldr      d23, [x23], #1             // encoding: [0xf7,0x16,0x40,0xfc]
// CHECK: ldr      d25, [x0], #-256           // encoding: [0x19,0x04,0x50,0xfc]

        ldr q20, [x1], #255
        ldr q23, [x9], #1
        ldr q25, [x20], #-256
        str q10, [x1], #255
        str q22, [sp], #1
        str q21, [x20], #-256
// CHECK: ldr      q20, [x1], #255            // encoding: [0x34,0xf4,0xcf,0x3c]
// CHECK: ldr      q23, [x9], #1              // encoding: [0x37,0x15,0xc0,0x3c]
// CHECK: ldr      q25, [x20], #-256          // encoding: [0x99,0x06,0xd0,0x3c]
// CHECK: str      q10, [x1], #255            // encoding: [0x2a,0xf4,0x8f,0x3c]
// CHECK: str      q22, [sp], #1              // encoding: [0xf6,0x17,0x80,0x3c]
// CHECK: str      q21, [x20], #-256          // encoding: [0x95,0x06,0x90,0x3c]

//------------------------------------------------------------------------------
// Load/store register (immediate pre-indexed)
//------------------------------------------------------------------------------

        ldr x3, [x4, #0]!
        ldr xzr, [sp, #0]!
// CHECK: ldr      x3, [x4, #0]!              // encoding: [0x83,0x0c,0x40,0xf8]
// CHECK: ldr      xzr, [sp, #0]!              // encoding: [0xff,0x0f,0x40,0xf8]

        strb w9, [x2, #255]!
        strb w10, [x3, #1]!
        strb w10, [x3, #-256]!
        strh w9, [x2, #255]!
        strh w9, [x2, #1]!
        strh w10, [x3, #-256]!
// CHECK: strb     w9, [x2, #255]!            // encoding: [0x49,0xfc,0x0f,0x38]
// CHECK: strb     w10, [x3, #1]!             // encoding: [0x6a,0x1c,0x00,0x38]
// CHECK: strb     w10, [x3, #-256]!          // encoding: [0x6a,0x0c,0x10,0x38]
// CHECK: strh     w9, [x2, #255]!            // encoding: [0x49,0xfc,0x0f,0x78]
// CHECK: strh     w9, [x2, #1]!              // encoding: [0x49,0x1c,0x00,0x78]
// CHECK: strh     w10, [x3, #-256]!          // encoding: [0x6a,0x0c,0x10,0x78]

        str w19, [sp, #255]!
        str w20, [x30, #1]!
        str w21, [x12, #-256]!
        str xzr, [x9, #255]!
        str x2, [x3, #1]!
        str x19, [x12, #-256]!
// CHECK: str      w19, [sp, #255]!           // encoding: [0xf3,0xff,0x0f,0xb8]
// CHECK: str      w20, [x30, #1]!            // encoding: [0xd4,0x1f,0x00,0xb8]
// CHECK: str      w21, [x12, #-256]!         // encoding: [0x95,0x0d,0x10,0xb8]
// CHECK: str      xzr, [x9, #255]!           // encoding: [0x3f,0xfd,0x0f,0xf8]
// CHECK: str      x2, [x3, #1]!              // encoding: [0x62,0x1c,0x00,0xf8]
// CHECK: str      x19, [x12, #-256]!         // encoding: [0x93,0x0d,0x10,0xf8]

        ldrb w9, [x2, #255]!
        ldrb w10, [x3, #1]!
        ldrb w10, [x3, #-256]!
        ldrh w9, [x2, #255]!
        ldrh w9, [x2, #1]!
        ldrh w10, [x3, #-256]!
// CHECK: ldrb     w9, [x2, #255]!            // encoding: [0x49,0xfc,0x4f,0x38]
// CHECK: ldrb     w10, [x3, #1]!             // encoding: [0x6a,0x1c,0x40,0x38]
// CHECK: ldrb     w10, [x3, #-256]!          // encoding: [0x6a,0x0c,0x50,0x38]
// CHECK: ldrh     w9, [x2, #255]!            // encoding: [0x49,0xfc,0x4f,0x78]
// CHECK: ldrh     w9, [x2, #1]!              // encoding: [0x49,0x1c,0x40,0x78]
// CHECK: ldrh     w10, [x3, #-256]!          // encoding: [0x6a,0x0c,0x50,0x78]

        ldr w19, [sp, #255]!
        ldr w20, [x30, #1]!
        ldr w21, [x12, #-256]!
        ldr xzr, [x9, #255]!
        ldr x2, [x3, #1]!
        ldr x19, [x12, #-256]!
// CHECK: ldr      w19, [sp, #255]!           // encoding: [0xf3,0xff,0x4f,0xb8]
// CHECK: ldr      w20, [x30, #1]!            // encoding: [0xd4,0x1f,0x40,0xb8]
// CHECK: ldr      w21, [x12, #-256]!         // encoding: [0x95,0x0d,0x50,0xb8]
// CHECK: ldr      xzr, [x9, #255]!           // encoding: [0x3f,0xfd,0x4f,0xf8]
// CHECK: ldr      x2, [x3, #1]!              // encoding: [0x62,0x1c,0x40,0xf8]
// CHECK: ldr      x19, [x12, #-256]!         // encoding: [0x93,0x0d,0x50,0xf8]

        ldrsb xzr, [x9, #255]!
        ldrsb x2, [x3, #1]!
        ldrsb x19, [x12, #-256]!
        ldrsh xzr, [x9, #255]!
        ldrsh x2, [x3, #1]!
        ldrsh x19, [x12, #-256]!
        ldrsw xzr, [x9, #255]!
        ldrsw x2, [x3, #1]!
        ldrsw x19, [x12, #-256]!
// CHECK: ldrsb    xzr, [x9, #255]!           // encoding: [0x3f,0xfd,0x8f,0x38]
// CHECK: ldrsb    x2, [x3, #1]!              // encoding: [0x62,0x1c,0x80,0x38]
// CHECK: ldrsb    x19, [x12, #-256]!         // encoding: [0x93,0x0d,0x90,0x38]
// CHECK: ldrsh    xzr, [x9, #255]!           // encoding: [0x3f,0xfd,0x8f,0x78]
// CHECK: ldrsh    x2, [x3, #1]!              // encoding: [0x62,0x1c,0x80,0x78]
// CHECK: ldrsh    x19, [x12, #-256]!         // encoding: [0x93,0x0d,0x90,0x78]
// CHECK: ldrsw    xzr, [x9, #255]!           // encoding: [0x3f,0xfd,0x8f,0xb8]
// CHECK: ldrsw    x2, [x3, #1]!              // encoding: [0x62,0x1c,0x80,0xb8]
// CHECK: ldrsw    x19, [x12, #-256]!         // encoding: [0x93,0x0d,0x90,0xb8]

        ldrsb wzr, [x9, #255]!
        ldrsb w2, [x3, #1]!
        ldrsb w19, [x12, #-256]!
        ldrsh wzr, [x9, #255]!
        ldrsh w2, [x3, #1]!
        ldrsh w19, [x12, #-256]!
// CHECK: ldrsb    wzr, [x9, #255]!           // encoding: [0x3f,0xfd,0xcf,0x38]
// CHECK: ldrsb    w2, [x3, #1]!              // encoding: [0x62,0x1c,0xc0,0x38]
// CHECK: ldrsb    w19, [x12, #-256]!         // encoding: [0x93,0x0d,0xd0,0x38]
// CHECK: ldrsh    wzr, [x9, #255]!           // encoding: [0x3f,0xfd,0xcf,0x78]
// CHECK: ldrsh    w2, [x3, #1]!              // encoding: [0x62,0x1c,0xc0,0x78]
// CHECK: ldrsh    w19, [x12, #-256]!         // encoding: [0x93,0x0d,0xd0,0x78]

        str b0, [x0, #255]!
        str b3, [x3, #1]!
        str b5, [sp, #-256]!
        str h10, [x10, #255]!
        str h13, [x23, #1]!
        str h15, [sp, #-256]!
        str s20, [x20, #255]!
        str s23, [x23, #1]!
        str s25, [x0, #-256]!
        str d20, [x20, #255]!
        str d23, [x23, #1]!
        str d25, [x0, #-256]!
// CHECK: str      b0, [x0, #255]!            // encoding: [0x00,0xfc,0x0f,0x3c]
// CHECK: str      b3, [x3, #1]!              // encoding: [0x63,0x1c,0x00,0x3c]
// CHECK: str      b5, [sp, #-256]!           // encoding: [0xe5,0x0f,0x10,0x3c]
// CHECK: str      h10, [x10, #255]!          // encoding: [0x4a,0xfd,0x0f,0x7c]
// CHECK: str      h13, [x23, #1]!            // encoding: [0xed,0x1e,0x00,0x7c]
// CHECK: str      h15, [sp, #-256]!          // encoding: [0xef,0x0f,0x10,0x7c]
// CHECK: str      s20, [x20, #255]!          // encoding: [0x94,0xfe,0x0f,0xbc]
// CHECK: str      s23, [x23, #1]!            // encoding: [0xf7,0x1e,0x00,0xbc]
// CHECK: str      s25, [x0, #-256]!          // encoding: [0x19,0x0c,0x10,0xbc]
// CHECK: str      d20, [x20, #255]!          // encoding: [0x94,0xfe,0x0f,0xfc]
// CHECK: str      d23, [x23, #1]!            // encoding: [0xf7,0x1e,0x00,0xfc]
// CHECK: str      d25, [x0, #-256]!          // encoding: [0x19,0x0c,0x10,0xfc]

        ldr b0, [x0, #255]!
        ldr b3, [x3, #1]!
        ldr b5, [sp, #-256]!
        ldr h10, [x10, #255]!
        ldr h13, [x23, #1]!
        ldr h15, [sp, #-256]!
        ldr s20, [x20, #255]!
        ldr s23, [x23, #1]!
        ldr s25, [x0, #-256]!
        ldr d20, [x20, #255]!
        ldr d23, [x23, #1]!
        ldr d25, [x0, #-256]!
// CHECK: ldr      b0, [x0, #255]!            // encoding: [0x00,0xfc,0x4f,0x3c]
// CHECK: ldr      b3, [x3, #1]!              // encoding: [0x63,0x1c,0x40,0x3c]
// CHECK: ldr      b5, [sp, #-256]!           // encoding: [0xe5,0x0f,0x50,0x3c]
// CHECK: ldr      h10, [x10, #255]!          // encoding: [0x4a,0xfd,0x4f,0x7c]
// CHECK: ldr      h13, [x23, #1]!            // encoding: [0xed,0x1e,0x40,0x7c]
// CHECK: ldr      h15, [sp, #-256]!          // encoding: [0xef,0x0f,0x50,0x7c]
// CHECK: ldr      s20, [x20, #255]!          // encoding: [0x94,0xfe,0x4f,0xbc]
// CHECK: ldr      s23, [x23, #1]!            // encoding: [0xf7,0x1e,0x40,0xbc]
// CHECK: ldr      s25, [x0, #-256]!          // encoding: [0x19,0x0c,0x50,0xbc]
// CHECK: ldr      d20, [x20, #255]!          // encoding: [0x94,0xfe,0x4f,0xfc]
// CHECK: ldr      d23, [x23, #1]!            // encoding: [0xf7,0x1e,0x40,0xfc]
// CHECK: ldr      d25, [x0, #-256]!          // encoding: [0x19,0x0c,0x50,0xfc]

        ldr q20, [x1, #255]!
        ldr q23, [x9, #1]!
        ldr q25, [x20, #-256]!
        str q10, [x1, #255]!
        str q22, [sp, #1]!
        str q21, [x20, #-256]!
// CHECK: ldr      q20, [x1, #255]!           // encoding: [0x34,0xfc,0xcf,0x3c]
// CHECK: ldr      q23, [x9, #1]!             // encoding: [0x37,0x1d,0xc0,0x3c]
// CHECK: ldr      q25, [x20, #-256]!         // encoding: [0x99,0x0e,0xd0,0x3c]
// CHECK: str      q10, [x1, #255]!           // encoding: [0x2a,0xfc,0x8f,0x3c]
// CHECK: str      q22, [sp, #1]!             // encoding: [0xf6,0x1f,0x80,0x3c]
// CHECK: str      q21, [x20, #-256]!         // encoding: [0x95,0x0e,0x90,0x3c]

//------------------------------------------------------------------------------
// Load/store (unprivileged)
//------------------------------------------------------------------------------

        sttrb w9, [sp, #0]
        sttrh wzr, [x12, #255]
        sttr w16, [x0, #-256]
        sttr x28, [x14, #1]
// CHECK: sttrb    w9, [sp]                   // encoding: [0xe9,0x0b,0x00,0x38]
// CHECK: sttrh    wzr, [x12, #255]           // encoding: [0x9f,0xf9,0x0f,0x78]
// CHECK: sttr     w16, [x0, #-256]           // encoding: [0x10,0x08,0x10,0xb8]
// CHECK: sttr     x28, [x14, #1]             // encoding: [0xdc,0x19,0x00,0xf8]

        ldtrb w1, [x20, #255]
        ldtrh w20, [x1, #255]
        ldtr w12, [sp, #255]
        ldtr xzr, [x12, #255]
// CHECK: ldtrb    w1, [x20, #255]            // encoding: [0x81,0xfa,0x4f,0x38]
// CHECK: ldtrh    w20, [x1, #255]            // encoding: [0x34,0xf8,0x4f,0x78]
// CHECK: ldtr     w12, [sp, #255]            // encoding: [0xec,0xfb,0x4f,0xb8]
// CHECK: ldtr     xzr, [x12, #255]           // encoding: [0x9f,0xf9,0x4f,0xf8]

        ldtrsb x9, [x7, #-256]
        ldtrsh x17, [x19, #-256]
        ldtrsw x20, [x15, #-256]
        ldtrsb w19, [x1, #-256]
        ldtrsh w15, [x21, #-256]
// CHECK: ldtrsb   x9, [x7, #-256]            // encoding: [0xe9,0x08,0x90,0x38]
// CHECK: ldtrsh   x17, [x19, #-256]          // encoding: [0x71,0x0a,0x90,0x78]
// CHECK: ldtrsw   x20, [x15, #-256]          // encoding: [0xf4,0x09,0x90,0xb8]
// CHECK: ldtrsb   w19, [x1, #-256]           // encoding: [0x33,0x08,0xd0,0x38]
// CHECK: ldtrsh   w15, [x21, #-256]          // encoding: [0xaf,0x0a,0xd0,0x78]

//------------------------------------------------------------------------------
// Load/store register pair (offset)
//------------------------------------------------------------------------------

        ldp w3, w5, [sp]
        stp wzr, w9, [sp, #252]
        ldp w2, wzr, [sp, #-256]
        ldp w9, w10, [sp, #4]
// CHECK: ldp      w3, w5, [sp]               // encoding: [0xe3,0x17,0x40,0x29]
// CHECK: stp      wzr, w9, [sp, #252]        // encoding: [0xff,0xa7,0x1f,0x29]
// CHECK: ldp      w2, wzr, [sp, #-256]       // encoding: [0xe2,0x7f,0x60,0x29]
// CHECK: ldp      w9, w10, [sp, #4]          // encoding: [0xe9,0xab,0x40,0x29]

        ldpsw x9, x10, [sp, #4]
        ldpsw x9, x10, [x2, #-256]
        ldpsw x20, x30, [sp, #252]
// CHECK: ldpsw    x9, x10, [sp, #4]          // encoding: [0xe9,0xab,0x40,0x69]
// CHECK: ldpsw    x9, x10, [x2, #-256]       // encoding: [0x49,0x28,0x60,0x69]
// CHECK: ldpsw    x20, x30, [sp, #252]       // encoding: [0xf4,0xfb,0x5f,0x69]

        ldp x21, x29, [x2, #504]
        ldp x22, x23, [x3, #-512]
        ldp x24, x25, [x4, #8]
// CHECK: ldp      x21, x29, [x2, #504]       // encoding: [0x55,0xf4,0x5f,0xa9]
// CHECK: ldp      x22, x23, [x3, #-512]      // encoding: [0x76,0x5c,0x60,0xa9]
// CHECK: ldp      x24, x25, [x4, #8]         // encoding: [0x98,0xe4,0x40,0xa9]

        ldp s29, s28, [sp, #252]
        stp s27, s26, [sp, #-256]
        ldp s1, s2, [x3, #44]
// CHECK: ldp      s29, s28, [sp, #252]       // encoding: [0xfd,0xf3,0x5f,0x2d]
// CHECK: stp      s27, s26, [sp, #-256]      // encoding: [0xfb,0x6b,0x20,0x2d]
// CHECK: ldp      s1, s2, [x3, #44]          // encoding: [0x61,0x88,0x45,0x2d]

        stp d3, d5, [x9, #504]
        stp d7, d11, [x10, #-512]
        ldp d2, d3, [x30, #-8]
// CHECK: stp      d3, d5, [x9, #504]         // encoding: [0x23,0x95,0x1f,0x6d]
// CHECK: stp      d7, d11, [x10, #-512]      // encoding: [0x47,0x2d,0x20,0x6d]
// CHECK: ldp      d2, d3, [x30, #-8]         // encoding: [0xc2,0x8f,0x7f,0x6d]

        stp q3, q5, [sp]
        stp q17, q19, [sp, #1008]
        ldp q23, q29, [x1, #-1024]
// CHECK: stp      q3, q5, [sp]               // encoding: [0xe3,0x17,0x00,0xad]
// CHECK: stp      q17, q19, [sp, #1008]      // encoding: [0xf1,0xcf,0x1f,0xad]
// CHECK: ldp      q23, q29, [x1, #-1024]     // encoding: [0x37,0x74,0x60,0xad]

//------------------------------------------------------------------------------
// Load/store register pair (post-indexed)
//------------------------------------------------------------------------------

        ldp w3, w5, [sp], #0
        stp wzr, w9, [sp], #252
        ldp w2, wzr, [sp], #-256
        ldp w9, w10, [sp], #4
// CHECK: ldp      w3, w5, [sp], #0           // encoding: [0xe3,0x17,0xc0,0x28]
// CHECK: stp      wzr, w9, [sp], #252        // encoding: [0xff,0xa7,0x9f,0x28]
// CHECK: ldp      w2, wzr, [sp], #-256       // encoding: [0xe2,0x7f,0xe0,0x28]
// CHECK: ldp      w9, w10, [sp], #4          // encoding: [0xe9,0xab,0xc0,0x28]

        ldpsw x9, x10, [sp], #4
        ldpsw x9, x10, [x2], #-256
        ldpsw x20, x30, [sp], #252
// CHECK: ldpsw    x9, x10, [sp], #4          // encoding: [0xe9,0xab,0xc0,0x68]
// CHECK: ldpsw    x9, x10, [x2], #-256       // encoding: [0x49,0x28,0xe0,0x68]
// CHECK: ldpsw    x20, x30, [sp], #252       // encoding: [0xf4,0xfb,0xdf,0x68]

        ldp x21, x29, [x2], #504
        ldp x22, x23, [x3], #-512
        ldp x24, x25, [x4], #8
// CHECK: ldp      x21, x29, [x2], #504       // encoding: [0x55,0xf4,0xdf,0xa8]
// CHECK: ldp      x22, x23, [x3], #-512      // encoding: [0x76,0x5c,0xe0,0xa8]
// CHECK: ldp      x24, x25, [x4], #8         // encoding: [0x98,0xe4,0xc0,0xa8]

        ldp s29, s28, [sp], #252
        stp s27, s26, [sp], #-256
        ldp s1, s2, [x3], #44
// CHECK: ldp      s29, s28, [sp], #252       // encoding: [0xfd,0xf3,0xdf,0x2c]
// CHECK: stp      s27, s26, [sp], #-256      // encoding: [0xfb,0x6b,0xa0,0x2c]
// CHECK: ldp      s1, s2, [x3], #44          // encoding: [0x61,0x88,0xc5,0x2c]

        stp d3, d5, [x9], #504
        stp d7, d11, [x10], #-512
        ldp d2, d3, [x30], #-8
// CHECK: stp      d3, d5, [x9], #504         // encoding: [0x23,0x95,0x9f,0x6c]
// CHECK: stp      d7, d11, [x10], #-512      // encoding: [0x47,0x2d,0xa0,0x6c]
// CHECK: ldp      d2, d3, [x30], #-8         // encoding: [0xc2,0x8f,0xff,0x6c]

        stp q3, q5, [sp], #0
        stp q17, q19, [sp], #1008
        ldp q23, q29, [x1], #-1024
// CHECK: stp      q3, q5, [sp], #0           // encoding: [0xe3,0x17,0x80,0xac]
// CHECK: stp      q17, q19, [sp], #1008      // encoding: [0xf1,0xcf,0x9f,0xac]
// CHECK: ldp      q23, q29, [x1], #-1024     // encoding: [0x37,0x74,0xe0,0xac]

//------------------------------------------------------------------------------
// Load/store register pair (pre-indexed)
//------------------------------------------------------------------------------
        ldp w3, w5, [sp, #0]!
        stp wzr, w9, [sp, #252]!
        ldp w2, wzr, [sp, #-256]!
        ldp w9, w10, [sp, #4]!
// CHECK: ldp      w3, w5, [sp, #0]!          // encoding: [0xe3,0x17,0xc0,0x29]
// CHECK: stp      wzr, w9, [sp, #252]!       // encoding: [0xff,0xa7,0x9f,0x29]
// CHECK: ldp      w2, wzr, [sp, #-256]!      // encoding: [0xe2,0x7f,0xe0,0x29]
// CHECK: ldp      w9, w10, [sp, #4]!         // encoding: [0xe9,0xab,0xc0,0x29]

        ldpsw x9, x10, [sp, #4]!
        ldpsw x9, x10, [x2, #-256]!
        ldpsw x20, x30, [sp, #252]!
// CHECK: ldpsw    x9, x10, [sp, #4]!         // encoding: [0xe9,0xab,0xc0,0x69]
// CHECK: ldpsw    x9, x10, [x2, #-256]!      // encoding: [0x49,0x28,0xe0,0x69]
// CHECK: ldpsw    x20, x30, [sp, #252]!      // encoding: [0xf4,0xfb,0xdf,0x69]

        ldp x21, x29, [x2, #504]!
        ldp x22, x23, [x3, #-512]!
        ldp x24, x25, [x4, #8]!
// CHECK: ldp      x21, x29, [x2, #504]!      // encoding: [0x55,0xf4,0xdf,0xa9]
// CHECK: ldp      x22, x23, [x3, #-512]!     // encoding: [0x76,0x5c,0xe0,0xa9]
// CHECK: ldp      x24, x25, [x4, #8]!        // encoding: [0x98,0xe4,0xc0,0xa9]

        ldp s29, s28, [sp, #252]!
        stp s27, s26, [sp, #-256]!
        ldp s1, s2, [x3, #44]!
// CHECK: ldp      s29, s28, [sp, #252]!      // encoding: [0xfd,0xf3,0xdf,0x2d]
// CHECK: stp      s27, s26, [sp, #-256]!     // encoding: [0xfb,0x6b,0xa0,0x2d]
// CHECK: ldp      s1, s2, [x3, #44]!         // encoding: [0x61,0x88,0xc5,0x2d]

        stp d3, d5, [x9, #504]!
        stp d7, d11, [x10, #-512]!
        ldp d2, d3, [x30, #-8]!
// CHECK: stp      d3, d5, [x9, #504]!        // encoding: [0x23,0x95,0x9f,0x6d]
// CHECK: stp      d7, d11, [x10, #-512]!     // encoding: [0x47,0x2d,0xa0,0x6d]
// CHECK: ldp      d2, d3, [x30, #-8]!        // encoding: [0xc2,0x8f,0xff,0x6d]

        stp q3, q5, [sp, #0]!
        stp q17, q19, [sp, #1008]!
        ldp q23, q29, [x1, #-1024]!
// CHECK: stp      q3, q5, [sp, #0]!          // encoding: [0xe3,0x17,0x80,0xad]
// CHECK: stp      q17, q19, [sp, #1008]!     // encoding: [0xf1,0xcf,0x9f,0xad]
// CHECK: ldp      q23, q29, [x1, #-1024]!    // encoding: [0x37,0x74,0xe0,0xad]

//------------------------------------------------------------------------------
// Load/store non-temporal register pair (offset)
//------------------------------------------------------------------------------

        ldnp w3, w5, [sp]
        stnp wzr, w9, [sp, #252]
        ldnp w2, wzr, [sp, #-256]
        ldnp w9, w10, [sp, #4]
// CHECK: ldnp      w3, w5, [sp]               // encoding: [0xe3,0x17,0x40,0x28]
// CHECK: stnp      wzr, w9, [sp, #252]        // encoding: [0xff,0xa7,0x1f,0x28]
// CHECK: ldnp      w2, wzr, [sp, #-256]       // encoding: [0xe2,0x7f,0x60,0x28]
// CHECK: ldnp      w9, w10, [sp, #4]          // encoding: [0xe9,0xab,0x40,0x28]

        ldnp x21, x29, [x2, #504]
        ldnp x22, x23, [x3, #-512]
        ldnp x24, x25, [x4, #8]
// CHECK: ldnp      x21, x29, [x2, #504]       // encoding: [0x55,0xf4,0x5f,0xa8]
// CHECK: ldnp      x22, x23, [x3, #-512]      // encoding: [0x76,0x5c,0x60,0xa8]
// CHECK: ldnp      x24, x25, [x4, #8]         // encoding: [0x98,0xe4,0x40,0xa8]

        ldnp s29, s28, [sp, #252]
        stnp s27, s26, [sp, #-256]
        ldnp s1, s2, [x3, #44]
// CHECK: ldnp      s29, s28, [sp, #252]       // encoding: [0xfd,0xf3,0x5f,0x2c]
// CHECK: stnp      s27, s26, [sp, #-256]      // encoding: [0xfb,0x6b,0x20,0x2c]
// CHECK: ldnp      s1, s2, [x3, #44]          // encoding: [0x61,0x88,0x45,0x2c]

        stnp d3, d5, [x9, #504]
        stnp d7, d11, [x10, #-512]
        ldnp d2, d3, [x30, #-8]
// CHECK: stnp      d3, d5, [x9, #504]         // encoding: [0x23,0x95,0x1f,0x6c]
// CHECK: stnp      d7, d11, [x10, #-512]      // encoding: [0x47,0x2d,0x20,0x6c]
// CHECK: ldnp      d2, d3, [x30, #-8]         // encoding: [0xc2,0x8f,0x7f,0x6c]

        stnp q3, q5, [sp]
        stnp q17, q19, [sp, #1008]
        ldnp q23, q29, [x1, #-1024]
// CHECK: stnp      q3, q5, [sp]               // encoding: [0xe3,0x17,0x00,0xac]
// CHECK: stnp      q17, q19, [sp, #1008]      // encoding: [0xf1,0xcf,0x1f,0xac]
// CHECK: ldnp      q23, q29, [x1, #-1024]     // encoding: [0x37,0x74,0x60,0xac]

//------------------------------------------------------------------------------
// Logical (immediate)
//------------------------------------------------------------------------------
        // 32 bit replication-width
        orr w3, w9, #0xffff0000
        orr wsp, w10, #0xe00000ff
        orr w9, w10, #0x000003ff
// CHECK: orr      w3, w9, #0xffff0000        // encoding: [0x23,0x3d,0x10,0x32]
// CHECK: orr      wsp, w10, #0xe00000ff      // encoding: [0x5f,0x29,0x03,0x32]
// CHECK: orr      w9, w10, #0x3ff            // encoding: [0x49,0x25,0x00,0x32]

        // 16 bit replication width
        and w14, w15, #0x80008000
        and w12, w13, #0xffc3ffc3
        and w11, wzr, #0x00030003
// CHECK: and      w14, w15, #0x80008000      // encoding: [0xee,0x81,0x01,0x12]
// CHECK: and      w12, w13, #0xffc3ffc3      // encoding: [0xac,0xad,0x0a,0x12]
// CHECK: and      w11, wzr, #0x30003         // encoding: [0xeb,0x87,0x00,0x12]

        // 8 bit replication width
        eor w3, w6, #0xe0e0e0e0
        eor wsp, wzr, #0x03030303
        eor w16, w17, #0x81818181
// CHECK: eor      w3, w6, #0xe0e0e0e0        // encoding: [0xc3,0xc8,0x03,0x52]
// CHECK: eor      wsp, wzr, #0x3030303       // encoding: [0xff,0xc7,0x00,0x52]
// CHECK: eor      w16, w17, #0x81818181      // encoding: [0x30,0xc6,0x01,0x52]

        // 4 bit replication width
        ands wzr, w18, #0xcccccccc
        ands w19, w20, #0x33333333
        ands w21, w22, #0x99999999
// CHECK: {{ands wzr,|tst}} w18, #0xcccccccc      // encoding: [0x5f,0xe6,0x02,0x72]
// CHECK: ands     w19, w20, #0x33333333      // encoding: [0x93,0xe6,0x00,0x72]
// CHECK: ands     w21, w22, #0x99999999      // encoding: [0xd5,0xe6,0x01,0x72]

        // 2 bit replication width
        tst w3, #0xaaaaaaaa
        tst wzr, #0x55555555
// CHECK: {{ands wzr,|tst}} w3, #0xaaaaaaaa       // encoding: [0x7f,0xf0,0x01,0x72]
// CHECK: {{ands wzr,|tst}} wzr, #0x55555555      // encoding: [0xff,0xf3,0x00,0x72]

        // 64 bit replication-width
        eor x3, x5, #0xffffffffc000000
        and x9, x10, #0x00007fffffffffff
        orr x11, x12, #0x8000000000000fff
// CHECK: eor      x3, x5, #0xffffffffc000000 // encoding: [0xa3,0x84,0x66,0xd2]
// CHECK: and      x9, x10, #0x7fffffffffff   // encoding: [0x49,0xb9,0x40,0x92]
// CHECK: orr      x11, x12, #0x8000000000000fff // encoding: [0x8b,0x31,0x41,0xb2]

        // 32 bit replication-width
        orr x3, x9, #0xffff0000ffff0000
        orr sp, x10, #0xe00000ffe00000ff
        orr x9, x10, #0x000003ff000003ff
// CHECK: orr      x3, x9, #0xffff0000ffff0000 // encoding: [0x23,0x3d,0x10,0xb2]
// CHECK: orr      sp, x10, #0xe00000ffe00000ff // encoding: [0x5f,0x29,0x03,0xb2]
// CHECK: orr      x9, x10, #0x3ff000003ff    // encoding: [0x49,0x25,0x00,0xb2]

        // 16 bit replication-width
        and x14, x15, #0x8000800080008000
        and x12, x13, #0xffc3ffc3ffc3ffc3
        and x11, xzr, #0x0003000300030003
// CHECK: and      x14, x15, #0x8000800080008000 // encoding: [0xee,0x81,0x01,0x92]
// CHECK: and      x12, x13, #0xffc3ffc3ffc3ffc3 // encoding: [0xac,0xad,0x0a,0x92]
// CHECK: and      x11, xzr, #0x3000300030003 // encoding: [0xeb,0x87,0x00,0x92]

        // 8 bit replication-width
        eor x3, x6, #0xe0e0e0e0e0e0e0e0
        eor sp, xzr, #0x0303030303030303
        eor x16, x17, #0x8181818181818181
// CHECK: eor      x3, x6, #0xe0e0e0e0e0e0e0e0 // encoding: [0xc3,0xc8,0x03,0xd2]
// CHECK: eor      sp, xzr, #0x303030303030303 // encoding: [0xff,0xc7,0x00,0xd2]
// CHECK: eor      x16, x17, #0x8181818181818181 // encoding: [0x30,0xc6,0x01,0xd2]

        // 4 bit replication-width
        ands xzr, x18, #0xcccccccccccccccc
        ands x19, x20, #0x3333333333333333
        ands x21, x22, #0x9999999999999999
// CHECK: {{ands xzr,|tst}} x18, #0xcccccccccccccccc // encoding: [0x5f,0xe6,0x02,0xf2]
// CHECK: ands     x19, x20, #0x3333333333333333 // encoding: [0x93,0xe6,0x00,0xf2]
// CHECK: ands     x21, x22, #0x9999999999999999 // encoding: [0xd5,0xe6,0x01,0xf2]

        // 2 bit replication-width
        tst x3, #0xaaaaaaaaaaaaaaaa
        tst xzr, #0x5555555555555555
// CHECK: {{ands xzr,|tst}} x3, #0xaaaaaaaaaaaaaaaa    // encoding: [0x7f,0xf0,0x01,0xf2]
// CHECK: {{ands xzr,|tst}} xzr, #0x5555555555555555   // encoding: [0xff,0xf3,0x00,0xf2]

        mov w3, #0xf000f
        mov x10, #0xaaaaaaaaaaaaaaaa
// CHECK: mov w3, #983055                // encoding: [0xe3,0x8f,0x00,0x32]
// CHECK: mov x10, #-6148914691236517206 // encoding: [0xea,0xf3,0x01,0xb2]

        // The Imm field of logicalImm operations has to be truncated to the
        // register width, i.e. 32 bits
        and w2, w3, #-3
        orr w0, w1, #~2
        eor w16, w17, #-7
        ands w19, w20, #~15
// CHECK: and	w2, w3, #0xfffffffd     // encoding: [0x62,0x78,0x1e,0x12]
// CHECK: orr	w0, w1, #0xfffffffd     // encoding: [0x20,0x78,0x1e,0x32]
// CHECK: eor	w16, w17, #0xfffffff9   // encoding: [0x30,0x76,0x1d,0x52]
// CHECK: ands	w19, w20, #0xfffffff0   // encoding: [0x93,0x6e,0x1c,0x72]

//------------------------------------------------------------------------------
// Logical (shifted register)
//------------------------------------------------------------------------------

        and w12, w23, w21
        and w16, w15, w1, lsl #1
        and w9, w4, w10, lsl #31
        and w3, w30, w11, lsl #0
        and x3, x5, x7, lsl #63
// CHECK: and      w12, w23, w21              // encoding: [0xec,0x02,0x15,0x0a]
// CHECK: and      w16, w15, w1, lsl #1       // encoding: [0xf0,0x05,0x01,0x0a]
// CHECK: and      w9, w4, w10, lsl #31       // encoding: [0x89,0x7c,0x0a,0x0a]
// CHECK: and      w3, w30, w11               // encoding: [0xc3,0x03,0x0b,0x0a]
// CHECK: and      x3, x5, x7, lsl #63        // encoding: [0xa3,0xfc,0x07,0x8a]

        and x5, x14, x19, asr #4
        and w3, w17, w19, ror #31
        and w0, w2, wzr, lsr #17
        and w3, w30, w11, asr #0
// CHECK: and      x5, x14, x19, asr #4       // encoding: [0xc5,0x11,0x93,0x8a]
// CHECK: and      w3, w17, w19, ror #31      // encoding: [0x23,0x7e,0xd3,0x0a]
// CHECK: and      w0, w2, wzr, lsr #17       // encoding: [0x40,0x44,0x5f,0x0a]
// CHECK: and      w3, w30, w11, asr #0       // encoding: [0xc3,0x03,0x8b,0x0a]

        and xzr, x4, x26, lsl #0
        and w3, wzr, w20, ror #0
        and x7, x20, xzr, asr #63
// CHECK: and      xzr, x4, x26               // encoding: [0x9f,0x00,0x1a,0x8a]
// CHECK: and      w3, wzr, w20, ror #0       // encoding: [0xe3,0x03,0xd4,0x0a]
// CHECK: and      x7, x20, xzr, asr #63      // encoding: [0x87,0xfe,0x9f,0x8a]

        bic x13, x20, x14, lsl #47
        bic w2, w7, w9
        orr w2, w7, w0, asr #31
        orr x8, x9, x10, lsl #12
        orn x3, x5, x7, asr #0
        orn w2, w5, w29
// CHECK: bic      x13, x20, x14, lsl #47     // encoding: [0x8d,0xbe,0x2e,0x8a]
// CHECK: bic      w2, w7, w9                 // encoding: [0xe2,0x00,0x29,0x0a]
// CHECK: orr      w2, w7, w0, asr #31        // encoding: [0xe2,0x7c,0x80,0x2a]
// CHECK: orr      x8, x9, x10, lsl #12       // encoding: [0x28,0x31,0x0a,0xaa]
// CHECK: orn      x3, x5, x7, asr #0         // encoding: [0xa3,0x00,0xa7,0xaa]
// CHECK: orn      w2, w5, w29                // encoding: [0xa2,0x00,0x3d,0x2a]

        ands w7, wzr, w9, lsl #1
        ands x3, x5, x20, ror #63
        bics w3, w5, w7, lsl #0
        bics x3, xzr, x3, lsl #1
// CHECK: ands     w7, wzr, w9, lsl #1        // encoding: [0xe7,0x07,0x09,0x6a]
// CHECK: ands     x3, x5, x20, ror #63       // encoding: [0xa3,0xfc,0xd4,0xea]
// CHECK: bics     w3, w5, w7                 // encoding: [0xa3,0x00,0x27,0x6a]
// CHECK: bics     x3, xzr, x3, lsl #1        // encoding: [0xe3,0x07,0x23,0xea]

        tst w3, w7, lsl #31
        tst x2, x20, asr #0
// CHECK: tst      w3, w7, lsl #31            // encoding: [0x7f,0x7c,0x07,0x6a]
// CHECK: tst      x2, x20, asr #0            // encoding: [0x5f,0x00,0x94,0xea]

        mov x3, x6
        mov x3, xzr
        mov wzr, w2
        mov w3, w5
// CHECK: mov      x3, x6                     // encoding: [0xe3,0x03,0x06,0xaa]
// CHECK: mov      x3, xzr                    // encoding: [0xe3,0x03,0x1f,0xaa]
// CHECK: mov      wzr, w2                    // encoding: [0xff,0x03,0x02,0x2a]
// CHECK: mov      w3, w5                     // encoding: [0xe3,0x03,0x05,0x2a]

//------------------------------------------------------------------------------
// Move wide (immediate)
//------------------------------------------------------------------------------

        movz w1, #65535, lsl #0
        movz w2, #0, lsl #16
        movn w2, #1234, lsl #0
// CHECK: mov     w1, #65535                  // encoding: [0xe1,0xff,0x9f,0x52]
// CHECK: movz     w2, #0, lsl #16            // encoding: [0x02,0x00,0xa0,0x52]
// CHECK: mov     w2, #-1235                  // encoding: [0x42,0x9a,0x80,0x12]

        movz x2, #1234, lsl #32
        movk xzr, #4321, lsl #48
// CHECK: mov      x2, #5299989643264             // encoding: [0x42,0x9a,0xc0,0xd2]
// CHECK: movk     xzr, #{{4321|0x10e1}}, lsl #48 // encoding: [0x3f,0x1c,0xe2,0xf2]

        movz x2, #:abs_g0:sym
        movk w3, #:abs_g0_nc:sym

// CHECK: movz    x2, #:abs_g0:sym        // encoding: [0bAAA00010,A,0b100AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g0:sym, kind: fixup_aarch64_movw
// CHECK: movk    w3, #:abs_g0_nc:sym     // encoding: [0bAAA00011,A,0b100AAAAA,0x72]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g0_nc:sym, kind: fixup_aarch64_movw

        movz x4, #:abs_g1:sym
        movk w5, #:abs_g1_nc:sym

// CHECK: movz    x4, #:abs_g1:sym        // encoding: [0bAAA00100,A,0b101AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g1:sym, kind: fixup_aarch64_movw
// CHECK: movk    w5, #:abs_g1_nc:sym     // encoding: [0bAAA00101,A,0b101AAAAA,0x72]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g1_nc:sym, kind: fixup_aarch64_movw

        movz x6, #:abs_g2:sym
        movk x7, #:abs_g2_nc:sym

// CHECK: movz    x6, #:abs_g2:sym        // encoding: [0bAAA00110,A,0b110AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g2:sym, kind: fixup_aarch64_movw
// CHECK: movk    x7, #:abs_g2_nc:sym     // encoding: [0bAAA00111,A,0b110AAAAA,0xf2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g2_nc:sym, kind: fixup_aarch64_movw

        movz x8, #:abs_g3:sym
        movk x9, #:abs_g3:sym

// CHECK: movz    x8, #:abs_g3:sym        // encoding: [0bAAA01000,A,0b111AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g3:sym, kind: fixup_aarch64_movw
// CHECK: movk    x9, #:abs_g3:sym        // encoding: [0bAAA01001,A,0b111AAAAA,0xf2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g3:sym, kind: fixup_aarch64_movw


        movn x30, #:abs_g0_s:sym
        movz x19, #:abs_g0_s:sym
        movn w10, #:abs_g0_s:sym
        movz w25, #:abs_g0_s:sym

// CHECK: movn    x30, #:abs_g0_s:sym     // encoding: [0bAAA11110,A,0b100AAAAA,0x92]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g0_s:sym, kind: fixup_aarch64_movw
// CHECK: movz    x19, #:abs_g0_s:sym     // encoding: [0bAAA10011,A,0b100AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g0_s:sym, kind: fixup_aarch64_movw
// CHECK: movn    w10, #:abs_g0_s:sym     // encoding: [0bAAA01010,A,0b100AAAAA,0x12]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g0_s:sym, kind: fixup_aarch64_movw
// CHECK: movz    w25, #:abs_g0_s:sym     // encoding: [0bAAA11001,A,0b100AAAAA,0x52]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g0_s:sym, kind: fixup_aarch64_movw

        movn x30, #:abs_g1_s:sym
        movz x19, #:abs_g1_s:sym
        movn w10, #:abs_g1_s:sym
        movz w25, #:abs_g1_s:sym

// CHECK: movn    x30, #:abs_g1_s:sym     // encoding: [0bAAA11110,A,0b101AAAAA,0x92]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g1_s:sym, kind: fixup_aarch64_movw
// CHECK: movz    x19, #:abs_g1_s:sym     // encoding: [0bAAA10011,A,0b101AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g1_s:sym, kind: fixup_aarch64_movw
// CHECK: movn    w10, #:abs_g1_s:sym     // encoding: [0bAAA01010,A,0b101AAAAA,0x12]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g1_s:sym, kind: fixup_aarch64_movw
// CHECK: movz    w25, #:abs_g1_s:sym     // encoding: [0bAAA11001,A,0b101AAAAA,0x52]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g1_s:sym, kind: fixup_aarch64_movw

        movn x30, #:abs_g2_s:sym
        movz x19, #:abs_g2_s:sym

// CHECK: movn    x30, #:abs_g2_s:sym     // encoding: [0bAAA11110,A,0b110AAAAA,0x92]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g2_s:sym, kind: fixup_aarch64_movw
// CHECK: movz    x19, #:abs_g2_s:sym     // encoding: [0bAAA10011,A,0b110AAAAA,0xd2]
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :abs_g2_s:sym, kind: fixup_aarch64_movw

//------------------------------------------------------------------------------
// PC-relative addressing
//------------------------------------------------------------------------------

        adr x2, loc
        adr xzr, loc

// CHECK: adr    x2, loc                 // encoding: [0x02'A',A,A,0x10'A']
// CHECK:                                 //   fixup A - offset: 0, value: loc, kind: fixup_aarch64_pcrel_adr_imm21
// CHECK: adr    xzr, loc                // encoding: [0x1f'A',A,A,0x10'A']
// CHECK:                                 //   fixup A - offset: 0, value: loc, kind: fixup_aarch64_pcrel_adr_imm21

        adrp x29, loc

// CHECK: adrp    x29, loc                // encoding: [0x1d'A',A,A,0x90'A']
// CHECK:                                 //   fixup A - offset: 0, value: loc, kind: fixup_aarch64_pcrel_adrp_imm21
        adrp x30, #4096
        adr x20, #0
        adr x9, #-1
        adr x5, #1048575
// CHECK: adrp    x30, #4096              // encoding: [0x1e,0x00,0x00,0xb0]
// CHECK: adr     x20, #0                 // encoding: [0x14,0x00,0x00,0x10]
// CHECK: adr     x9, #-1                 // encoding: [0xe9,0xff,0xff,0x70]
// CHECK: adr     x5, #1048575            // encoding: [0xe5,0xff,0x7f,0x70]

        adr x9, #1048575
        adr x2, #-1048576
        adrp x9, #4294963200
        adrp x20, #-4294967296
// CHECK: adr     x9, #1048575            // encoding: [0xe9,0xff,0x7f,0x70]
// CHECK: adr     x2, #-1048576           // encoding: [0x02,0x00,0x80,0x10]
// CHECK: adrp    x9, #4294963200         // encoding: [0xe9,0xff,0x7f,0xf0]
// CHECK: adrp    x20, #-4294967296       // encoding: [0x14,0x00,0x80,0x90]

//------------------------------------------------------------------------------
// System
//------------------------------------------------------------------------------

        hint #0
        hint #127
// CHECK: nop                             // encoding: [0x1f,0x20,0x03,0xd5]
// CHECK: hint    #{{127|0x7f}}           // encoding: [0xff,0x2f,0x03,0xd5]

        nop
        yield
        wfe
        wfi
        sev
        sevl
        dgh
// CHECK: nop                             // encoding: [0x1f,0x20,0x03,0xd5]
// CHECK: yield                           // encoding: [0x3f,0x20,0x03,0xd5]
// CHECK: wfe                             // encoding: [0x5f,0x20,0x03,0xd5]
// CHECK: wfi                             // encoding: [0x7f,0x20,0x03,0xd5]
// CHECK: sev                             // encoding: [0x9f,0x20,0x03,0xd5]
// CHECK: sevl                            // encoding: [0xbf,0x20,0x03,0xd5]
// CHECK: dgh                             // encoding: [0xdf,0x20,0x03,0xd5]

        clrex
        clrex #0
        clrex #7
        clrex #15
// CHECK: clrex                           // encoding: [0x5f,0x3f,0x03,0xd5]
// CHECK: clrex   #0                      // encoding: [0x5f,0x30,0x03,0xd5]
// CHECK: clrex   #7                      // encoding: [0x5f,0x37,0x03,0xd5]
// CHECK: clrex                           // encoding: [0x5f,0x3f,0x03,0xd5]

        dsb #0
        dsb #4
        dsb #12
        dsb #15
        dsb oshld
        dsb oshst
        dsb osh
        dsb nshld
        dsb nshst
        dsb nsh
        dsb ishld
        dsb ishst
        dsb ish
        dsb ld
        dsb st
        dsb sy
// CHECK: ssbb                            // encoding: [0x9f,0x30,0x03,0xd5]
// CHECK: pssbb                           // encoding: [0x9f,0x34,0x03,0xd5]
// CHECK: dsb     #12                     // encoding: [0x9f,0x3c,0x03,0xd5]
// CHECK: dsb     sy                      // encoding: [0x9f,0x3f,0x03,0xd5]
// CHECK: dsb     oshld                   // encoding: [0x9f,0x31,0x03,0xd5]
// CHECK: dsb     oshst                   // encoding: [0x9f,0x32,0x03,0xd5]
// CHECK: dsb     osh                     // encoding: [0x9f,0x33,0x03,0xd5]
// CHECK: dsb     nshld                   // encoding: [0x9f,0x35,0x03,0xd5]
// CHECK: dsb     nshst                   // encoding: [0x9f,0x36,0x03,0xd5]
// CHECK: dsb     nsh                     // encoding: [0x9f,0x37,0x03,0xd5]
// CHECK: dsb     ishld                   // encoding: [0x9f,0x39,0x03,0xd5]
// CHECK: dsb     ishst                   // encoding: [0x9f,0x3a,0x03,0xd5]
// CHECK: dsb     ish                     // encoding: [0x9f,0x3b,0x03,0xd5]
// CHECK: dsb     ld                      // encoding: [0x9f,0x3d,0x03,0xd5]
// CHECK: dsb     st                      // encoding: [0x9f,0x3e,0x03,0xd5]
// CHECK: dsb     sy                      // encoding: [0x9f,0x3f,0x03,0xd5]

        dmb #0
        dmb #12
        dmb #15
        dmb oshld
        dmb oshst
        dmb osh
        dmb nshld
        dmb nshst
        dmb nsh
        dmb ishld
        dmb ishst
        dmb ish
        dmb ld
        dmb st
        dmb sy
// CHECK: dmb     #0                      // encoding: [0xbf,0x30,0x03,0xd5]
// CHECK: dmb     #12                     // encoding: [0xbf,0x3c,0x03,0xd5]
// CHECK: dmb     sy                      // encoding: [0xbf,0x3f,0x03,0xd5]
// CHECK: dmb     oshld                   // encoding: [0xbf,0x31,0x03,0xd5]
// CHECK: dmb     oshst                   // encoding: [0xbf,0x32,0x03,0xd5]
// CHECK: dmb     osh                     // encoding: [0xbf,0x33,0x03,0xd5]
// CHECK: dmb     nshld                   // encoding: [0xbf,0x35,0x03,0xd5]
// CHECK: dmb     nshst                   // encoding: [0xbf,0x36,0x03,0xd5]
// CHECK: dmb     nsh                     // encoding: [0xbf,0x37,0x03,0xd5]
// CHECK: dmb     ishld                   // encoding: [0xbf,0x39,0x03,0xd5]
// CHECK: dmb     ishst                   // encoding: [0xbf,0x3a,0x03,0xd5]
// CHECK: dmb     ish                     // encoding: [0xbf,0x3b,0x03,0xd5]
// CHECK: dmb     ld                      // encoding: [0xbf,0x3d,0x03,0xd5]
// CHECK: dmb     st                      // encoding: [0xbf,0x3e,0x03,0xd5]
// CHECK: dmb     sy                      // encoding: [0xbf,0x3f,0x03,0xd5]

        isb sy
        isb
        isb #12
// CHECK: isb                             // encoding: [0xdf,0x3f,0x03,0xd5]
// CHECK: isb                             // encoding: [0xdf,0x3f,0x03,0xd5]
// CHECK: isb     #12                     // encoding: [0xdf,0x3c,0x03,0xd5]


        msr spsel, #0
        msr daifset, #15
        msr daifclr, #12
// CHECK: msr     {{SPSel|SPSEL}}, #0               // encoding: [0xbf,0x40,0x00,0xd5]
// CHECK: msr     {{DAIFSet|DAIFSET}}, #15            // encoding: [0xdf,0x4f,0x03,0xd5]
// CHECK: msr     {{DAIFClr|DAIFCLR}}, #12            // encoding: [0xff,0x4c,0x03,0xd5]

        sys #7, c5, c9, #7, x5
        sys #0, c15, c15, #2
// CHECK: sys     #7, c5, c9, #7, x5      // encoding: [0xe5,0x59,0x0f,0xd5]
// CHECK: sys     #0, c15, c15, #2   // encoding: [0x5f,0xff,0x08,0xd5]

        sysl x9, #7, c5, c9, #7
        sysl x1, #0, c15, c15, #2
// CHECK: sysl    x9, #7, c5, c9, #7      // encoding: [0xe9,0x59,0x2f,0xd5]
// CHECK: sysl    x1, #0, c15, c15, #2    // encoding: [0x41,0xff,0x28,0xd5]

        ic ialluis
        ic iallu
        ic ivau, x9
// CHECK:         ic      ialluis                 // encoding: [0x1f,0x71,0x08,0xd5]
// CHECK:         ic      iallu                   // encoding: [0x1f,0x75,0x08,0xd5]
// CHECK:         ic      ivau, x9                // encoding: [0x29,0x75,0x0b,0xd5]

        dc zva, x12
        dc ivac, xzr
        dc isw, x2
        dc cvac, x9
        dc csw, x10
        dc cvau, x0
        dc civac, x3
        dc cisw, x30
// CHECK:         dc      zva, x12                // encoding: [0x2c,0x74,0x0b,0xd5]
// CHECK:         dc      ivac, xzr               // encoding: [0x3f,0x76,0x08,0xd5]
// CHECK:         dc      isw, x2                 // encoding: [0x42,0x76,0x08,0xd5]
// CHECK:         dc      cvac, x9                // encoding: [0x29,0x7a,0x0b,0xd5]
// CHECK:         dc      csw, x10                // encoding: [0x4a,0x7a,0x08,0xd5]
// CHECK:         dc      cvau, x0                // encoding: [0x20,0x7b,0x0b,0xd5]
// CHECK:         dc      civac, x3               // encoding: [0x23,0x7e,0x0b,0xd5]
// CHECK:         dc      cisw, x30               // encoding: [0x5e,0x7e,0x08,0xd5]

        at S1E1R, x19
        at S1E2R, x19
        at S1E3R, x19
        at S1E1W, x19
        at S1E2W, x19
        at S1E3W, x19
        at S1E0R, x19
        at S1E0W, x19
        at S12E1R, x20
        at S12E1W, x20
        at S12E0R, x20
        at S12E0W, x20
// CHECK: at      s1e1r, x19              // encoding: [0x13,0x78,0x08,0xd5]
// CHECK: at      s1e2r, x19              // encoding: [0x13,0x78,0x0c,0xd5]
// CHECK: at      s1e3r, x19              // encoding: [0x13,0x78,0x0e,0xd5]
// CHECK: at      s1e1w, x19              // encoding: [0x33,0x78,0x08,0xd5]
// CHECK: at      s1e2w, x19              // encoding: [0x33,0x78,0x0c,0xd5]
// CHECK: at      s1e3w, x19              // encoding: [0x33,0x78,0x0e,0xd5]
// CHECK: at      s1e0r, x19              // encoding: [0x53,0x78,0x08,0xd5]
// CHECK: at      s1e0w, x19              // encoding: [0x73,0x78,0x08,0xd5]
// CHECK: at      s12e1r, x20             // encoding: [0x94,0x78,0x0c,0xd5]
// CHECK: at      s12e1w, x20             // encoding: [0xb4,0x78,0x0c,0xd5]
// CHECK: at      s12e0r, x20             // encoding: [0xd4,0x78,0x0c,0xd5]
// CHECK: at      s12e0w, x20             // encoding: [0xf4,0x78,0x0c,0xd5]

        tlbi IPAS2E1IS, x4
        tlbi IPAS2LE1IS, x9
        tlbi VMALLE1IS
        tlbi ALLE2IS
        tlbi ALLE3IS
        tlbi VAE1IS, x1
        tlbi VAE2IS, x2
        tlbi VAE3IS, x3
        tlbi ASIDE1IS, x5
        tlbi VAAE1IS, x9
        tlbi ALLE1IS
        tlbi VALE1IS, x10
        tlbi VALE2IS, x11
        tlbi VALE3IS, x13
        tlbi VMALLS12E1IS
        tlbi VAALE1IS, x14
        tlbi IPAS2E1, x15
        tlbi IPAS2LE1, x16
        tlbi VMALLE1
        tlbi ALLE2
        tlbi ALLE3
        tlbi VAE1, x17
        tlbi VAE2, x18
        tlbi VAE3, x19
        tlbi ASIDE1, x20
        tlbi VAAE1, x21
        tlbi ALLE1
        tlbi VALE1, x22
        tlbi VALE2, x23
        tlbi VALE3, x24
        tlbi VMALLS12E1
        tlbi VAALE1, x25
// CHECK: tlbi    ipas2e1is, x4           // encoding: [0x24,0x80,0x0c,0xd5]
// CHECK: tlbi    ipas2le1is, x9          // encoding: [0xa9,0x80,0x0c,0xd5]
// CHECK: tlbi    vmalle1is               // encoding: [0x1f,0x83,0x08,0xd5]
// CHECK: tlbi    alle2is                 // encoding: [0x1f,0x83,0x0c,0xd5]
// CHECK: tlbi    alle3is                 // encoding: [0x1f,0x83,0x0e,0xd5]
// CHECK: tlbi    vae1is, x1              // encoding: [0x21,0x83,0x08,0xd5]
// CHECK: tlbi    vae2is, x2              // encoding: [0x22,0x83,0x0c,0xd5]
// CHECK: tlbi    vae3is, x3              // encoding: [0x23,0x83,0x0e,0xd5]
// CHECK: tlbi    aside1is, x5            // encoding: [0x45,0x83,0x08,0xd5]
// CHECK: tlbi    vaae1is, x9             // encoding: [0x69,0x83,0x08,0xd5]
// CHECK: tlbi    alle1is                 // encoding: [0x9f,0x83,0x0c,0xd5]
// CHECK: tlbi    vale1is, x10            // encoding: [0xaa,0x83,0x08,0xd5]
// CHECK: tlbi    vale2is, x11            // encoding: [0xab,0x83,0x0c,0xd5]
// CHECK: tlbi    vale3is, x13            // encoding: [0xad,0x83,0x0e,0xd5]
// CHECK: tlbi    vmalls12e1is            // encoding: [0xdf,0x83,0x0c,0xd5]
// CHECK: tlbi    vaale1is, x14           // encoding: [0xee,0x83,0x08,0xd5]
// CHECK: tlbi    ipas2e1, x15            // encoding: [0x2f,0x84,0x0c,0xd5]
// CHECK: tlbi    ipas2le1, x16           // encoding: [0xb0,0x84,0x0c,0xd5]
// CHECK: tlbi    vmalle1                 // encoding: [0x1f,0x87,0x08,0xd5]
// CHECK: tlbi    alle2                   // encoding: [0x1f,0x87,0x0c,0xd5]
// CHECK: tlbi    alle3                   // encoding: [0x1f,0x87,0x0e,0xd5]
// CHECK: tlbi    vae1, x17               // encoding: [0x31,0x87,0x08,0xd5]
// CHECK: tlbi    vae2, x18               // encoding: [0x32,0x87,0x0c,0xd5]
// CHECK: tlbi    vae3, x19               // encoding: [0x33,0x87,0x0e,0xd5]
// CHECK: tlbi    aside1, x20             // encoding: [0x54,0x87,0x08,0xd5]
// CHECK: tlbi    vaae1, x21              // encoding: [0x75,0x87,0x08,0xd5]
// CHECK: tlbi    alle1                   // encoding: [0x9f,0x87,0x0c,0xd5]
// CHECK: tlbi    vale1, x22              // encoding: [0xb6,0x87,0x08,0xd5]
// CHECK: tlbi    vale2, x23              // encoding: [0xb7,0x87,0x0c,0xd5]
// CHECK: tlbi    vale3, x24              // encoding: [0xb8,0x87,0x0e,0xd5]
// CHECK: tlbi    vmalls12e1              // encoding: [0xdf,0x87,0x0c,0xd5]
// CHECK: tlbi    vaale1, x25             // encoding: [0xf9,0x87,0x08,0xd5]

	msr TEECR32_EL1, x12
	msr OSDTRRX_EL1, x12
	msr MDCCINT_EL1, x12
	msr MDSCR_EL1, x12
	msr OSDTRTX_EL1, x12
	msr DBGDTR_EL0, x12
	msr DBGDTRTX_EL0, x12
	msr OSECCR_EL1, x12
	msr DBGVCR32_EL2, x12
	msr DBGBVR0_EL1, x12
	msr DBGBVR1_EL1, x12
	msr DBGBVR2_EL1, x12
	msr DBGBVR3_EL1, x12
	msr DBGBVR4_EL1, x12
	msr DBGBVR5_EL1, x12
	msr DBGBVR6_EL1, x12
	msr DBGBVR7_EL1, x12
	msr DBGBVR8_EL1, x12
	msr DBGBVR9_EL1, x12
	msr DBGBVR10_EL1, x12
	msr DBGBVR11_EL1, x12
	msr DBGBVR12_EL1, x12
	msr DBGBVR13_EL1, x12
	msr DBGBVR14_EL1, x12
	msr DBGBVR15_EL1, x12
	msr DBGBCR0_EL1, x12
	msr DBGBCR1_EL1, x12
	msr DBGBCR2_EL1, x12
	msr DBGBCR3_EL1, x12
	msr DBGBCR4_EL1, x12
	msr DBGBCR5_EL1, x12
	msr DBGBCR6_EL1, x12
	msr DBGBCR7_EL1, x12
	msr DBGBCR8_EL1, x12
	msr DBGBCR9_EL1, x12
	msr DBGBCR10_EL1, x12
	msr DBGBCR11_EL1, x12
	msr DBGBCR12_EL1, x12
	msr DBGBCR13_EL1, x12
	msr DBGBCR14_EL1, x12
	msr DBGBCR15_EL1, x12
	msr DBGWVR0_EL1, x12
	msr DBGWVR1_EL1, x12
	msr DBGWVR2_EL1, x12
	msr DBGWVR3_EL1, x12
	msr DBGWVR4_EL1, x12
	msr DBGWVR5_EL1, x12
	msr DBGWVR6_EL1, x12
	msr DBGWVR7_EL1, x12
	msr DBGWVR8_EL1, x12
	msr DBGWVR9_EL1, x12
	msr DBGWVR10_EL1, x12
	msr DBGWVR11_EL1, x12
	msr DBGWVR12_EL1, x12
	msr DBGWVR13_EL1, x12
	msr DBGWVR14_EL1, x12
	msr DBGWVR15_EL1, x12
	msr DBGWCR0_EL1, x12
	msr DBGWCR1_EL1, x12
	msr DBGWCR2_EL1, x12
	msr DBGWCR3_EL1, x12
	msr DBGWCR4_EL1, x12
	msr DBGWCR5_EL1, x12
	msr DBGWCR6_EL1, x12
	msr DBGWCR7_EL1, x12
	msr DBGWCR8_EL1, x12
	msr DBGWCR9_EL1, x12
	msr DBGWCR10_EL1, x12
	msr DBGWCR11_EL1, x12
	msr DBGWCR12_EL1, x12
	msr DBGWCR13_EL1, x12
	msr DBGWCR14_EL1, x12
	msr DBGWCR15_EL1, x12
	msr TEEHBR32_EL1, x12
	msr OSLAR_EL1, x12
	msr OSDLR_EL1, x12
	msr DBGPRCR_EL1, x12
	msr DBGCLAIMSET_EL1, x12
	msr DBGCLAIMCLR_EL1, x12
	msr CSSELR_EL1, x12
	msr VPIDR_EL2, x12
	msr VMPIDR_EL2, x12
	msr SCTLR_EL1, x12
	msr SCTLR_EL2, x12
	msr SCTLR_EL3, x12
	msr ACTLR_EL1, x12
	msr ACTLR_EL2, x12
	msr ACTLR_EL3, x12
	msr CPACR_EL1, x12
	msr HCR_EL2, x12
	msr SCR_EL3, x12
	msr MDCR_EL2, x12
	msr SDER32_EL3, x12
	msr CPTR_EL2, x12
	msr CPTR_EL3, x12
	msr HSTR_EL2, x12
	msr HACR_EL2, x12
	msr MDCR_EL3, x12
	msr TTBR0_EL1, x12
	msr TTBR0_EL2, x12
	msr TTBR0_EL3, x12
	msr TTBR1_EL1, x12
	msr TCR_EL1, x12
	msr TCR_EL2, x12
	msr TCR_EL3, x12
	msr VTTBR_EL2, x12
	msr VTCR_EL2, x12
	msr DACR32_EL2, x12
	msr SPSR_EL1, x12
	msr SPSR_EL2, x12
	msr SPSR_EL3, x12
	msr ELR_EL1, x12
	msr ELR_EL2, x12
	msr ELR_EL3, x12
	msr SP_EL0, x12
	msr SP_EL1, x12
	msr SP_EL2, x12
	msr SPSel, x12
	msr NZCV, x12
	msr DAIF, x12
	msr SPSR_irq, x12
	msr SPSR_abt, x12
	msr SPSR_und, x12
	msr SPSR_fiq, x12
	msr FPCR, x12
	msr FPSR, x12
	msr DSPSR_EL0, x12
	msr DLR_EL0, x12
	msr IFSR32_EL2, x12
	msr AFSR0_EL1, x12
	msr AFSR0_EL2, x12
	msr AFSR0_EL3, x12
	msr AFSR1_EL1, x12
	msr AFSR1_EL2, x12
	msr AFSR1_EL3, x12
	msr ESR_EL1, x12
	msr ESR_EL2, x12
	msr ESR_EL3, x12
	msr FPEXC32_EL2, x12
	msr FAR_EL1, x12
	msr FAR_EL2, x12
	msr FAR_EL3, x12
	msr HPFAR_EL2, x12
	msr PAR_EL1, x12
	msr PMCR_EL0, x12
	msr PMCNTENSET_EL0, x12
	msr PMCNTENCLR_EL0, x12
	msr PMOVSCLR_EL0, x12
	msr PMSELR_EL0, x12
	msr PMCCNTR_EL0, x12
	msr PMXEVTYPER_EL0, x12
	msr PMXEVCNTR_EL0, x12
	msr PMUSERENR_EL0, x12
	msr PMINTENSET_EL1, x12
	msr PMINTENCLR_EL1, x12
	msr PMOVSSET_EL0, x12
	msr MAIR_EL1, x12
	msr MAIR_EL2, x12
	msr MAIR_EL3, x12
	msr AMAIR_EL1, x12
	msr AMAIR_EL2, x12
	msr AMAIR_EL3, x12
	msr VBAR_EL1, x12
	msr VBAR_EL2, x12
	msr VBAR_EL3, x12
	msr RMR_EL1, x12
	msr RMR_EL2, x12
	msr RMR_EL3, x12
	msr CONTEXTIDR_EL1, x12
	msr TPIDR_EL0, x12
	msr TPIDR_EL2, x12
	msr TPIDR_EL3, x12
	msr TPIDRRO_EL0, x12
	msr TPIDR_EL1, x12
	msr CNTFRQ_EL0, x12
	msr CNTVOFF_EL2, x12
	msr CNTKCTL_EL1, x12
	msr CNTHCTL_EL2, x12
	msr CNTP_TVAL_EL0, x12
	msr CNTHP_TVAL_EL2, x12
	msr CNTPS_TVAL_EL1, x12
	msr CNTP_CTL_EL0, x12
	msr CNTHP_CTL_EL2, x12
	msr CNTPS_CTL_EL1, x12
	msr CNTP_CVAL_EL0, x12
	msr CNTHP_CVAL_EL2, x12
	msr CNTPS_CVAL_EL1, x12
	msr CNTV_TVAL_EL0, x12
	msr CNTV_CTL_EL0, x12
	msr CNTV_CVAL_EL0, x12
	msr PMEVCNTR0_EL0, x12
	msr PMEVCNTR1_EL0, x12
	msr PMEVCNTR2_EL0, x12
	msr PMEVCNTR3_EL0, x12
	msr PMEVCNTR4_EL0, x12
	msr PMEVCNTR5_EL0, x12
	msr PMEVCNTR6_EL0, x12
	msr PMEVCNTR7_EL0, x12
	msr PMEVCNTR8_EL0, x12
	msr PMEVCNTR9_EL0, x12
	msr PMEVCNTR10_EL0, x12
	msr PMEVCNTR11_EL0, x12
	msr PMEVCNTR12_EL0, x12
	msr PMEVCNTR13_EL0, x12
	msr PMEVCNTR14_EL0, x12
	msr PMEVCNTR15_EL0, x12
	msr PMEVCNTR16_EL0, x12
	msr PMEVCNTR17_EL0, x12
	msr PMEVCNTR18_EL0, x12
	msr PMEVCNTR19_EL0, x12
	msr PMEVCNTR20_EL0, x12
	msr PMEVCNTR21_EL0, x12
	msr PMEVCNTR22_EL0, x12
	msr PMEVCNTR23_EL0, x12
	msr PMEVCNTR24_EL0, x12
	msr PMEVCNTR25_EL0, x12
	msr PMEVCNTR26_EL0, x12
	msr PMEVCNTR27_EL0, x12
	msr PMEVCNTR28_EL0, x12
	msr PMEVCNTR29_EL0, x12
	msr PMEVCNTR30_EL0, x12
	msr PMCCFILTR_EL0, x12
	msr PMEVTYPER0_EL0, x12
	msr PMEVTYPER1_EL0, x12
	msr PMEVTYPER2_EL0, x12
	msr PMEVTYPER3_EL0, x12
	msr PMEVTYPER4_EL0, x12
	msr PMEVTYPER5_EL0, x12
	msr PMEVTYPER6_EL0, x12
	msr PMEVTYPER7_EL0, x12
	msr PMEVTYPER8_EL0, x12
	msr PMEVTYPER9_EL0, x12
	msr PMEVTYPER10_EL0, x12
	msr PMEVTYPER11_EL0, x12
	msr PMEVTYPER12_EL0, x12
	msr PMEVTYPER13_EL0, x12
	msr PMEVTYPER14_EL0, x12
	msr PMEVTYPER15_EL0, x12
	msr PMEVTYPER16_EL0, x12
	msr PMEVTYPER17_EL0, x12
	msr PMEVTYPER18_EL0, x12
	msr PMEVTYPER19_EL0, x12
	msr PMEVTYPER20_EL0, x12
	msr PMEVTYPER21_EL0, x12
	msr PMEVTYPER22_EL0, x12
	msr PMEVTYPER23_EL0, x12
	msr PMEVTYPER24_EL0, x12
	msr PMEVTYPER25_EL0, x12
	msr PMEVTYPER26_EL0, x12
	msr PMEVTYPER27_EL0, x12
	msr PMEVTYPER28_EL0, x12
	msr PMEVTYPER29_EL0, x12
	msr PMEVTYPER30_EL0, x12
        msr AMAIR2_EL1, x12
        msr AMAIR2_EL12, x12
        msr AMAIR2_EL2, x12
        msr AMAIR2_EL3, x12
        msr MAIR2_EL1, x12
        msr MAIR2_EL12, x12
        msr MAIR2_EL2, x12
        msr MAIR2_EL3, x12
        msr PIRE0_EL1, x12
        msr PIRE0_EL2, x12
        msr PIR_EL1, x12
        msr PIR_EL12, x12
        msr PIR_EL2, x12
        msr PIR_EL3, x12
        msr S2PIR_EL2, x12
        msr POR_EL0, x12
        msr POR_EL1, x12
        msr POR_EL12, x12
        msr POR_EL2, x12
        msr POR_EL3, x12
        msr S2POR_EL1, x12
        msr SCTLR2_EL1, x12
        msr SCTLR2_EL12, x12
        msr SCTLR2_EL2, x12
        msr TCR2_EL1, x12
        msr TCR2_EL12, x12
        msr TCR2_EL2, x12
// CHECK: msr      {{teecr32_el1|TEECR32_EL1}}, x12           // encoding: [0x0c,0x00,0x12,0xd5]
// CHECK: msr      {{osdtrrx_el1|OSDTRRX_EL1}}, x12           // encoding: [0x4c,0x00,0x10,0xd5]
// CHECK: msr      {{mdccint_el1|MDCCINT_EL1}}, x12           // encoding: [0x0c,0x02,0x10,0xd5]
// CHECK: msr      {{mdscr_el1|MDSCR_EL1}}, x12             // encoding: [0x4c,0x02,0x10,0xd5]
// CHECK: msr      {{osdtrtx_el1|OSDTRTX_EL1}}, x12           // encoding: [0x4c,0x03,0x10,0xd5]
// CHECK: msr      {{dbgdtr_el0|DBGDTR_EL0}}, x12            // encoding: [0x0c,0x04,0x13,0xd5]
// CHECK: msr      {{dbgdtrtx_el0|DBGDTRTX_EL0}}, x12          // encoding: [0x0c,0x05,0x13,0xd5]
// CHECK: msr      {{oseccr_el1|OSECCR_EL1}}, x12            // encoding: [0x4c,0x06,0x10,0xd5]
// CHECK: msr      {{dbgvcr32_el2|DBGVCR32_EL2}}, x12          // encoding: [0x0c,0x07,0x14,0xd5]
// CHECK: msr      {{dbgbvr0_el1|DBGBVR0_EL1}}, x12           // encoding: [0x8c,0x00,0x10,0xd5]
// CHECK: msr      {{dbgbvr1_el1|DBGBVR1_EL1}}, x12           // encoding: [0x8c,0x01,0x10,0xd5]
// CHECK: msr      {{dbgbvr2_el1|DBGBVR2_EL1}}, x12           // encoding: [0x8c,0x02,0x10,0xd5]
// CHECK: msr      {{dbgbvr3_el1|DBGBVR3_EL1}}, x12           // encoding: [0x8c,0x03,0x10,0xd5]
// CHECK: msr      {{dbgbvr4_el1|DBGBVR4_EL1}}, x12           // encoding: [0x8c,0x04,0x10,0xd5]
// CHECK: msr      {{dbgbvr5_el1|DBGBVR5_EL1}}, x12           // encoding: [0x8c,0x05,0x10,0xd5]
// CHECK: msr      {{dbgbvr6_el1|DBGBVR6_EL1}}, x12           // encoding: [0x8c,0x06,0x10,0xd5]
// CHECK: msr      {{dbgbvr7_el1|DBGBVR7_EL1}}, x12           // encoding: [0x8c,0x07,0x10,0xd5]
// CHECK: msr      {{dbgbvr8_el1|DBGBVR8_EL1}}, x12           // encoding: [0x8c,0x08,0x10,0xd5]
// CHECK: msr      {{dbgbvr9_el1|DBGBVR9_EL1}}, x12           // encoding: [0x8c,0x09,0x10,0xd5]
// CHECK: msr      {{dbgbvr10_el1|DBGBVR10_EL1}}, x12          // encoding: [0x8c,0x0a,0x10,0xd5]
// CHECK: msr      {{dbgbvr11_el1|DBGBVR11_EL1}}, x12          // encoding: [0x8c,0x0b,0x10,0xd5]
// CHECK: msr      {{dbgbvr12_el1|DBGBVR12_EL1}}, x12          // encoding: [0x8c,0x0c,0x10,0xd5]
// CHECK: msr      {{dbgbvr13_el1|DBGBVR13_EL1}}, x12          // encoding: [0x8c,0x0d,0x10,0xd5]
// CHECK: msr      {{dbgbvr14_el1|DBGBVR14_EL1}}, x12          // encoding: [0x8c,0x0e,0x10,0xd5]
// CHECK: msr      {{dbgbvr15_el1|DBGBVR15_EL1}}, x12          // encoding: [0x8c,0x0f,0x10,0xd5]
// CHECK: msr      {{dbgbcr0_el1|DBGBCR0_EL1}}, x12           // encoding: [0xac,0x00,0x10,0xd5]
// CHECK: msr      {{dbgbcr1_el1|DBGBCR1_EL1}}, x12           // encoding: [0xac,0x01,0x10,0xd5]
// CHECK: msr      {{dbgbcr2_el1|DBGBCR2_EL1}}, x12           // encoding: [0xac,0x02,0x10,0xd5]
// CHECK: msr      {{dbgbcr3_el1|DBGBCR3_EL1}}, x12           // encoding: [0xac,0x03,0x10,0xd5]
// CHECK: msr      {{dbgbcr4_el1|DBGBCR4_EL1}}, x12           // encoding: [0xac,0x04,0x10,0xd5]
// CHECK: msr      {{dbgbcr5_el1|DBGBCR5_EL1}}, x12           // encoding: [0xac,0x05,0x10,0xd5]
// CHECK: msr      {{dbgbcr6_el1|DBGBCR6_EL1}}, x12           // encoding: [0xac,0x06,0x10,0xd5]
// CHECK: msr      {{dbgbcr7_el1|DBGBCR7_EL1}}, x12           // encoding: [0xac,0x07,0x10,0xd5]
// CHECK: msr      {{dbgbcr8_el1|DBGBCR8_EL1}}, x12           // encoding: [0xac,0x08,0x10,0xd5]
// CHECK: msr      {{dbgbcr9_el1|DBGBCR9_EL1}}, x12           // encoding: [0xac,0x09,0x10,0xd5]
// CHECK: msr      {{dbgbcr10_el1|DBGBCR10_EL1}}, x12          // encoding: [0xac,0x0a,0x10,0xd5]
// CHECK: msr      {{dbgbcr11_el1|DBGBCR11_EL1}}, x12          // encoding: [0xac,0x0b,0x10,0xd5]
// CHECK: msr      {{dbgbcr12_el1|DBGBCR12_EL1}}, x12          // encoding: [0xac,0x0c,0x10,0xd5]
// CHECK: msr      {{dbgbcr13_el1|DBGBCR13_EL1}}, x12          // encoding: [0xac,0x0d,0x10,0xd5]
// CHECK: msr      {{dbgbcr14_el1|DBGBCR14_EL1}}, x12          // encoding: [0xac,0x0e,0x10,0xd5]
// CHECK: msr      {{dbgbcr15_el1|DBGBCR15_EL1}}, x12          // encoding: [0xac,0x0f,0x10,0xd5]
// CHECK: msr      {{dbgwvr0_el1|DBGWVR0_EL1}}, x12           // encoding: [0xcc,0x00,0x10,0xd5]
// CHECK: msr      {{dbgwvr1_el1|DBGWVR1_EL1}}, x12           // encoding: [0xcc,0x01,0x10,0xd5]
// CHECK: msr      {{dbgwvr2_el1|DBGWVR2_EL1}}, x12           // encoding: [0xcc,0x02,0x10,0xd5]
// CHECK: msr      {{dbgwvr3_el1|DBGWVR3_EL1}}, x12           // encoding: [0xcc,0x03,0x10,0xd5]
// CHECK: msr      {{dbgwvr4_el1|DBGWVR4_EL1}}, x12           // encoding: [0xcc,0x04,0x10,0xd5]
// CHECK: msr      {{dbgwvr5_el1|DBGWVR5_EL1}}, x12           // encoding: [0xcc,0x05,0x10,0xd5]
// CHECK: msr      {{dbgwvr6_el1|DBGWVR6_EL1}}, x12           // encoding: [0xcc,0x06,0x10,0xd5]
// CHECK: msr      {{dbgwvr7_el1|DBGWVR7_EL1}}, x12           // encoding: [0xcc,0x07,0x10,0xd5]
// CHECK: msr      {{dbgwvr8_el1|DBGWVR8_EL1}}, x12           // encoding: [0xcc,0x08,0x10,0xd5]
// CHECK: msr      {{dbgwvr9_el1|DBGWVR9_EL1}}, x12           // encoding: [0xcc,0x09,0x10,0xd5]
// CHECK: msr      {{dbgwvr10_el1|DBGWVR10_EL1}}, x12          // encoding: [0xcc,0x0a,0x10,0xd5]
// CHECK: msr      {{dbgwvr11_el1|DBGWVR11_EL1}}, x12          // encoding: [0xcc,0x0b,0x10,0xd5]
// CHECK: msr      {{dbgwvr12_el1|DBGWVR12_EL1}}, x12          // encoding: [0xcc,0x0c,0x10,0xd5]
// CHECK: msr      {{dbgwvr13_el1|DBGWVR13_EL1}}, x12          // encoding: [0xcc,0x0d,0x10,0xd5]
// CHECK: msr      {{dbgwvr14_el1|DBGWVR14_EL1}}, x12          // encoding: [0xcc,0x0e,0x10,0xd5]
// CHECK: msr      {{dbgwvr15_el1|DBGWVR15_EL1}}, x12          // encoding: [0xcc,0x0f,0x10,0xd5]
// CHECK: msr      {{dbgwcr0_el1|DBGWCR0_EL1}}, x12           // encoding: [0xec,0x00,0x10,0xd5]
// CHECK: msr      {{dbgwcr1_el1|DBGWCR1_EL1}}, x12           // encoding: [0xec,0x01,0x10,0xd5]
// CHECK: msr      {{dbgwcr2_el1|DBGWCR2_EL1}}, x12           // encoding: [0xec,0x02,0x10,0xd5]
// CHECK: msr      {{dbgwcr3_el1|DBGWCR3_EL1}}, x12           // encoding: [0xec,0x03,0x10,0xd5]
// CHECK: msr      {{dbgwcr4_el1|DBGWCR4_EL1}}, x12           // encoding: [0xec,0x04,0x10,0xd5]
// CHECK: msr      {{dbgwcr5_el1|DBGWCR5_EL1}}, x12           // encoding: [0xec,0x05,0x10,0xd5]
// CHECK: msr      {{dbgwcr6_el1|DBGWCR6_EL1}}, x12           // encoding: [0xec,0x06,0x10,0xd5]
// CHECK: msr      {{dbgwcr7_el1|DBGWCR7_EL1}}, x12           // encoding: [0xec,0x07,0x10,0xd5]
// CHECK: msr      {{dbgwcr8_el1|DBGWCR8_EL1}}, x12           // encoding: [0xec,0x08,0x10,0xd5]
// CHECK: msr      {{dbgwcr9_el1|DBGWCR9_EL1}}, x12           // encoding: [0xec,0x09,0x10,0xd5]
// CHECK: msr      {{dbgwcr10_el1|DBGWCR10_EL1}}, x12          // encoding: [0xec,0x0a,0x10,0xd5]
// CHECK: msr      {{dbgwcr11_el1|DBGWCR11_EL1}}, x12          // encoding: [0xec,0x0b,0x10,0xd5]
// CHECK: msr      {{dbgwcr12_el1|DBGWCR12_EL1}}, x12          // encoding: [0xec,0x0c,0x10,0xd5]
// CHECK: msr      {{dbgwcr13_el1|DBGWCR13_EL1}}, x12          // encoding: [0xec,0x0d,0x10,0xd5]
// CHECK: msr      {{dbgwcr14_el1|DBGWCR14_EL1}}, x12          // encoding: [0xec,0x0e,0x10,0xd5]
// CHECK: msr      {{dbgwcr15_el1|DBGWCR15_EL1}}, x12          // encoding: [0xec,0x0f,0x10,0xd5]
// CHECK: msr      {{teehbr32_el1|TEEHBR32_EL1}}, x12          // encoding: [0x0c,0x10,0x12,0xd5]
// CHECK: msr      {{oslar_el1|OSLAR_EL1}}, x12             // encoding: [0x8c,0x10,0x10,0xd5]
// CHECK: msr      {{osdlr_el1|OSDLR_EL1}}, x12             // encoding: [0x8c,0x13,0x10,0xd5]
// CHECK: msr      {{dbgprcr_el1|DBGPRCR_EL1}}, x12           // encoding: [0x8c,0x14,0x10,0xd5]
// CHECK: msr      {{dbgclaimset_el1|DBGCLAIMSET_EL1}}, x12       // encoding: [0xcc,0x78,0x10,0xd5]
// CHECK: msr      {{dbgclaimclr_el1|DBGCLAIMCLR_EL1}}, x12       // encoding: [0xcc,0x79,0x10,0xd5]
// CHECK: msr      {{csselr_el1|CSSELR_EL1}}, x12            // encoding: [0x0c,0x00,0x1a,0xd5]
// CHECK: msr      {{vpidr_el2|VPIDR_EL2}}, x12             // encoding: [0x0c,0x00,0x1c,0xd5]
// CHECK: msr      {{vmpidr_el2|VMPIDR_EL2}}, x12            // encoding: [0xac,0x00,0x1c,0xd5]
// CHECK: msr      {{sctlr_el1|SCTLR_EL1}}, x12             // encoding: [0x0c,0x10,0x18,0xd5]
// CHECK: msr      {{sctlr_el2|SCTLR_EL2}}, x12             // encoding: [0x0c,0x10,0x1c,0xd5]
// CHECK: msr      {{sctlr_el3|SCTLR_EL3}}, x12             // encoding: [0x0c,0x10,0x1e,0xd5]
// CHECK: msr      {{actlr_el1|ACTLR_EL1}}, x12             // encoding: [0x2c,0x10,0x18,0xd5]
// CHECK: msr      {{actlr_el2|ACTLR_EL2}}, x12             // encoding: [0x2c,0x10,0x1c,0xd5]
// CHECK: msr      {{actlr_el3|ACTLR_EL3}}, x12             // encoding: [0x2c,0x10,0x1e,0xd5]
// CHECK: msr      {{cpacr_el1|CPACR_EL1}}, x12             // encoding: [0x4c,0x10,0x18,0xd5]
// CHECK: msr      {{hcr_el2|HCR_EL2}}, x12               // encoding: [0x0c,0x11,0x1c,0xd5]
// CHECK: msr      {{scr_el3|SCR_EL3}}, x12               // encoding: [0x0c,0x11,0x1e,0xd5]
// CHECK: msr      {{mdcr_el2|MDCR_EL2}}, x12              // encoding: [0x2c,0x11,0x1c,0xd5]
// CHECK: msr      {{sder32_el3|SDER32_EL3}}, x12            // encoding: [0x2c,0x11,0x1e,0xd5]
// CHECK: msr      {{cptr_el2|CPTR_EL2}}, x12              // encoding: [0x4c,0x11,0x1c,0xd5]
// CHECK: msr      {{cptr_el3|CPTR_EL3}}, x12              // encoding: [0x4c,0x11,0x1e,0xd5]
// CHECK: msr      {{hstr_el2|HSTR_EL2}}, x12              // encoding: [0x6c,0x11,0x1c,0xd5]
// CHECK: msr      {{hacr_el2|HACR_EL2}}, x12              // encoding: [0xec,0x11,0x1c,0xd5]
// CHECK: msr      {{mdcr_el3|MDCR_EL3}}, x12              // encoding: [0x2c,0x13,0x1e,0xd5]
// CHECK: msr      {{ttbr0_el1|TTBR0_EL1}}, x12             // encoding: [0x0c,0x20,0x18,0xd5]
// CHECK: msr      {{ttbr0_el2|TTBR0_EL2}}, x12             // encoding: [0x0c,0x20,0x1c,0xd5]
// CHECK: msr      {{ttbr0_el3|TTBR0_EL3}}, x12             // encoding: [0x0c,0x20,0x1e,0xd5]
// CHECK: msr      {{ttbr1_el1|TTBR1_EL1}}, x12             // encoding: [0x2c,0x20,0x18,0xd5]
// CHECK: msr      {{tcr_el1|TCR_EL1}}, x12               // encoding: [0x4c,0x20,0x18,0xd5]
// CHECK: msr      {{tcr_el2|TCR_EL2}}, x12               // encoding: [0x4c,0x20,0x1c,0xd5]
// CHECK: msr      {{tcr_el3|TCR_EL3}}, x12               // encoding: [0x4c,0x20,0x1e,0xd5]
// CHECK: msr      {{vttbr_el2|VTTBR_EL2}}, x12            // encoding: [0x0c,0x21,0x1c,0xd5]
// CHECK: msr      {{vtcr_el2|VTCR_EL2}}, x12              // encoding: [0x4c,0x21,0x1c,0xd5]
// CHECK: msr      {{dacr32_el2|DACR32_EL2}}, x12            // encoding: [0x0c,0x30,0x1c,0xd5]
// CHECK: msr      {{spsr_el1|SPSR_EL1}}, x12              // encoding: [0x0c,0x40,0x18,0xd5]
// CHECK: msr      {{spsr_el2|SPSR_EL2}}, x12              // encoding: [0x0c,0x40,0x1c,0xd5]
// CHECK: msr      {{spsr_el3|SPSR_EL3}}, x12              // encoding: [0x0c,0x40,0x1e,0xd5]
// CHECK: msr      {{elr_el1|ELR_EL1}}, x12               // encoding: [0x2c,0x40,0x18,0xd5]
// CHECK: msr      {{elr_el2|ELR_EL2}}, x12               // encoding: [0x2c,0x40,0x1c,0xd5]
// CHECK: msr      {{elr_el3|ELR_EL3}}, x12               // encoding: [0x2c,0x40,0x1e,0xd5]
// CHECK: msr      {{sp_el0|SP_EL0}}, x12                // encoding: [0x0c,0x41,0x18,0xd5]
// CHECK: msr      {{sp_el1|SP_EL1}}, x12                // encoding: [0x0c,0x41,0x1c,0xd5]
// CHECK: msr      {{sp_el2|SP_EL2}}, x12                // encoding: [0x0c,0x41,0x1e,0xd5]
// CHECK: msr      {{SPSel|SPSEL}}, x12                 // encoding: [0x0c,0x42,0x18,0xd5]
// CHECK: msr      {{nzcv|NZCV}}, x12                  // encoding: [0x0c,0x42,0x1b,0xd5]
// CHECK: msr      {{daif|DAIF}}, x12                  // encoding: [0x2c,0x42,0x1b,0xd5]
// CHECK: msr      {{SPSR_irq|SPSR_IRQ}}, x12              // encoding: [0x0c,0x43,0x1c,0xd5]
// CHECK: msr      {{SPSR_abt|SPSR_ABT}}, x12              // encoding: [0x2c,0x43,0x1c,0xd5]
// CHECK: msr      {{SPSR_und|SPSR_UND}}, x12              // encoding: [0x4c,0x43,0x1c,0xd5]
// CHECK: msr      {{SPSR_fiq|SPSR_FIQ}}, x12              // encoding: [0x6c,0x43,0x1c,0xd5]
// CHECK: msr      {{fpcr|FPCR}}, x12                  // encoding: [0x0c,0x44,0x1b,0xd5]
// CHECK: msr      {{fpsr|FPSR}}, x12                  // encoding: [0x2c,0x44,0x1b,0xd5]
// CHECK: msr      {{dspsr_el0|DSPSR_EL0}}, x12             // encoding: [0x0c,0x45,0x1b,0xd5]
// CHECK: msr      {{dlr_el0|DLR_EL0}}, x12               // encoding: [0x2c,0x45,0x1b,0xd5]
// CHECK: msr      {{ifsr32_el2|IFSR32_EL2}}, x12            // encoding: [0x2c,0x50,0x1c,0xd5]
// CHECK: msr      {{afsr0_el1|AFSR0_EL1}}, x12             // encoding: [0x0c,0x51,0x18,0xd5]
// CHECK: msr      {{afsr0_el2|AFSR0_EL2}}, x12             // encoding: [0x0c,0x51,0x1c,0xd5]
// CHECK: msr      {{afsr0_el3|AFSR0_EL3}}, x12             // encoding: [0x0c,0x51,0x1e,0xd5]
// CHECK: msr      {{afsr1_el1|AFSR1_EL1}}, x12             // encoding: [0x2c,0x51,0x18,0xd5]
// CHECK: msr      {{afsr1_el2|AFSR1_EL2}}, x12             // encoding: [0x2c,0x51,0x1c,0xd5]
// CHECK: msr      {{afsr1_el3|AFSR1_EL3}}, x12             // encoding: [0x2c,0x51,0x1e,0xd5]
// CHECK: msr      {{esr_el1|ESR_EL1}}, x12               // encoding: [0x0c,0x52,0x18,0xd5]
// CHECK: msr      {{esr_el2|ESR_EL2}}, x12               // encoding: [0x0c,0x52,0x1c,0xd5]
// CHECK: msr      {{esr_el3|ESR_EL3}}, x12               // encoding: [0x0c,0x52,0x1e,0xd5]
// CHECK: msr      {{fpexc32_el2|FPEXC32_EL2}}, x12           // encoding: [0x0c,0x53,0x1c,0xd5]
// CHECK: msr      {{far_el1|FAR_EL1}}, x12               // encoding: [0x0c,0x60,0x18,0xd5]
// CHECK: msr      {{far_el2|FAR_EL2}}, x12               // encoding: [0x0c,0x60,0x1c,0xd5]
// CHECK: msr      {{far_el3|FAR_EL3}}, x12               // encoding: [0x0c,0x60,0x1e,0xd5]
// CHECK: msr      {{hpfar_el2|HPFAR_EL2}}, x12             // encoding: [0x8c,0x60,0x1c,0xd5]
// CHECK: msr      {{par_el1|PAR_EL1}}, x12               // encoding: [0x0c,0x74,0x18,0xd5]
// CHECK: msr      {{pmcr_el0|PMCR_EL0}}, x12              // encoding: [0x0c,0x9c,0x1b,0xd5]
// CHECK: msr      {{pmcntenset_el0|PMCNTENSET_EL0}}, x12        // encoding: [0x2c,0x9c,0x1b,0xd5]
// CHECK: msr      {{pmcntenclr_el0|PMCNTENCLR_EL0}}, x12        // encoding: [0x4c,0x9c,0x1b,0xd5]
// CHECK: msr      {{pmovsclr_el0|PMOVSCLR_EL0}}, x12          // encoding: [0x6c,0x9c,0x1b,0xd5]
// CHECK: msr      {{pmselr_el0|PMSELR_EL0}}, x12            // encoding: [0xac,0x9c,0x1b,0xd5]
// CHECK: msr      {{pmccntr_el0|PMCCNTR_EL0}}, x12           // encoding: [0x0c,0x9d,0x1b,0xd5]
// CHECK: msr      {{pmxevtyper_el0|PMXEVTYPER_EL0}}, x12        // encoding: [0x2c,0x9d,0x1b,0xd5]
// CHECK: msr      {{pmxevcntr_el0|PMXEVCNTR_EL0}}, x12         // encoding: [0x4c,0x9d,0x1b,0xd5]
// CHECK: msr      {{pmuserenr_el0|PMUSERENR_EL0}}, x12         // encoding: [0x0c,0x9e,0x1b,0xd5]
// CHECK: msr      {{pmintenset_el1|PMINTENSET_EL1}}, x12        // encoding: [0x2c,0x9e,0x18,0xd5]
// CHECK: msr      {{pmintenclr_el1|PMINTENCLR_EL1}}, x12        // encoding: [0x4c,0x9e,0x18,0xd5]
// CHECK: msr      {{pmovsset_el0|PMOVSSET_EL0}}, x12          // encoding: [0x6c,0x9e,0x1b,0xd5]
// CHECK: msr      {{mair_el1|MAIR_EL1}}, x12              // encoding: [0x0c,0xa2,0x18,0xd5]
// CHECK: msr      {{mair_el2|MAIR_EL2}}, x12              // encoding: [0x0c,0xa2,0x1c,0xd5]
// CHECK: msr      {{mair_el3|MAIR_EL3}}, x12              // encoding: [0x0c,0xa2,0x1e,0xd5]
// CHECK: msr      {{amair_el1|AMAIR_EL1}}, x12             // encoding: [0x0c,0xa3,0x18,0xd5]
// CHECK: msr      {{amair_el2|AMAIR_EL2}}, x12             // encoding: [0x0c,0xa3,0x1c,0xd5]
// CHECK: msr      {{amair_el3|AMAIR_EL3}}, x12             // encoding: [0x0c,0xa3,0x1e,0xd5]
// CHECK: msr      {{vbar_el1|VBAR_EL1}}, x12              // encoding: [0x0c,0xc0,0x18,0xd5]
// CHECK: msr      {{vbar_el2|VBAR_EL2}}, x12              // encoding: [0x0c,0xc0,0x1c,0xd5]
// CHECK: msr      {{vbar_el3|VBAR_EL3}}, x12              // encoding: [0x0c,0xc0,0x1e,0xd5]
// CHECK: msr      {{rmr_el1|RMR_EL1}}, x12               // encoding: [0x4c,0xc0,0x18,0xd5]
// CHECK: msr      {{rmr_el2|RMR_EL2}}, x12               // encoding: [0x4c,0xc0,0x1c,0xd5]
// CHECK: msr      {{rmr_el3|RMR_EL3}}, x12               // encoding: [0x4c,0xc0,0x1e,0xd5]
// CHECK: msr      {{contextidr_el1|CONTEXTIDR_EL1}}, x12        // encoding: [0x2c,0xd0,0x18,0xd5]
// CHECK: msr      {{tpidr_el0|TPIDR_EL0}}, x12             // encoding: [0x4c,0xd0,0x1b,0xd5]
// CHECK: msr      {{tpidr_el2|TPIDR_EL2}}, x12             // encoding: [0x4c,0xd0,0x1c,0xd5]
// CHECK: msr      {{tpidr_el3|TPIDR_EL3}}, x12             // encoding: [0x4c,0xd0,0x1e,0xd5]
// CHECK: msr      {{tpidrro_el0|TPIDRRO_EL0}}, x12           // encoding: [0x6c,0xd0,0x1b,0xd5]
// CHECK: msr      {{tpidr_el1|TPIDR_EL1}}, x12             // encoding: [0x8c,0xd0,0x18,0xd5]
// CHECK: msr      {{cntfrq_el0|CNTFRQ_EL0}}, x12            // encoding: [0x0c,0xe0,0x1b,0xd5]
// CHECK: msr      {{cntvoff_el2|CNTVOFF_EL2}}, x12           // encoding: [0x6c,0xe0,0x1c,0xd5]
// CHECK: msr      {{cntkctl_el1|CNTKCTL_EL1}}, x12           // encoding: [0x0c,0xe1,0x18,0xd5]
// CHECK: msr      {{cnthctl_el2|CNTHCTL_EL2}}, x12           // encoding: [0x0c,0xe1,0x1c,0xd5]
// CHECK: msr      {{cntp_tval_el0|CNTP_TVAL_EL0}}, x12         // encoding: [0x0c,0xe2,0x1b,0xd5]
// CHECK: msr      {{cnthp_tval_el2|CNTHP_TVAL_EL2}}, x12        // encoding: [0x0c,0xe2,0x1c,0xd5]
// CHECK: msr      {{cntps_tval_el1|CNTPS_TVAL_EL1}}, x12        // encoding: [0x0c,0xe2,0x1f,0xd5]
// CHECK: msr      {{cntp_ctl_el0|CNTP_CTL_EL0}}, x12          // encoding: [0x2c,0xe2,0x1b,0xd5]
// CHECK: msr      {{cnthp_ctl_el2|CNTHP_CTL_EL2}}, x12         // encoding: [0x2c,0xe2,0x1c,0xd5]
// CHECK: msr      {{cntps_ctl_el1|CNTPS_CTL_EL1}}, x12         // encoding: [0x2c,0xe2,0x1f,0xd5]
// CHECK: msr      {{cntp_cval_el0|CNTP_CVAL_EL0}}, x12         // encoding: [0x4c,0xe2,0x1b,0xd5]
// CHECK: msr      {{cnthp_cval_el2|CNTHP_CVAL_EL2}}, x12        // encoding: [0x4c,0xe2,0x1c,0xd5]
// CHECK: msr      {{cntps_cval_el1|CNTPS_CVAL_EL1}}, x12        // encoding: [0x4c,0xe2,0x1f,0xd5]
// CHECK: msr      {{cntv_tval_el0|CNTV_TVAL_EL0}}, x12         // encoding: [0x0c,0xe3,0x1b,0xd5]
// CHECK: msr      {{cntv_ctl_el0|CNTV_CTL_EL0}}, x12          // encoding: [0x2c,0xe3,0x1b,0xd5]
// CHECK: msr      {{cntv_cval_el0|CNTV_CVAL_EL0}}, x12         // encoding: [0x4c,0xe3,0x1b,0xd5]
// CHECK: msr      {{pmevcntr0_el0|PMEVCNTR0_EL0}}, x12         // encoding: [0x0c,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr1_el0|PMEVCNTR1_EL0}}, x12         // encoding: [0x2c,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr2_el0|PMEVCNTR2_EL0}}, x12         // encoding: [0x4c,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr3_el0|PMEVCNTR3_EL0}}, x12         // encoding: [0x6c,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr4_el0|PMEVCNTR4_EL0}}, x12         // encoding: [0x8c,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr5_el0|PMEVCNTR5_EL0}}, x12         // encoding: [0xac,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr6_el0|PMEVCNTR6_EL0}}, x12         // encoding: [0xcc,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr7_el0|PMEVCNTR7_EL0}}, x12         // encoding: [0xec,0xe8,0x1b,0xd5]
// CHECK: msr      {{pmevcntr8_el0|PMEVCNTR8_EL0}}, x12         // encoding: [0x0c,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr9_el0|PMEVCNTR9_EL0}}, x12         // encoding: [0x2c,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr10_el0|PMEVCNTR10_EL0}}, x12        // encoding: [0x4c,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr11_el0|PMEVCNTR11_EL0}}, x12        // encoding: [0x6c,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr12_el0|PMEVCNTR12_EL0}}, x12        // encoding: [0x8c,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr13_el0|PMEVCNTR13_EL0}}, x12        // encoding: [0xac,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr14_el0|PMEVCNTR14_EL0}}, x12        // encoding: [0xcc,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr15_el0|PMEVCNTR15_EL0}}, x12        // encoding: [0xec,0xe9,0x1b,0xd5]
// CHECK: msr      {{pmevcntr16_el0|PMEVCNTR16_EL0}}, x12        // encoding: [0x0c,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr17_el0|PMEVCNTR17_EL0}}, x12        // encoding: [0x2c,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr18_el0|PMEVCNTR18_EL0}}, x12        // encoding: [0x4c,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr19_el0|PMEVCNTR19_EL0}}, x12        // encoding: [0x6c,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr20_el0|PMEVCNTR20_EL0}}, x12        // encoding: [0x8c,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr21_el0|PMEVCNTR21_EL0}}, x12        // encoding: [0xac,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr22_el0|PMEVCNTR22_EL0}}, x12        // encoding: [0xcc,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr23_el0|PMEVCNTR23_EL0}}, x12        // encoding: [0xec,0xea,0x1b,0xd5]
// CHECK: msr      {{pmevcntr24_el0|PMEVCNTR24_EL0}}, x12        // encoding: [0x0c,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmevcntr25_el0|PMEVCNTR25_EL0}}, x12        // encoding: [0x2c,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmevcntr26_el0|PMEVCNTR26_EL0}}, x12        // encoding: [0x4c,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmevcntr27_el0|PMEVCNTR27_EL0}}, x12        // encoding: [0x6c,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmevcntr28_el0|PMEVCNTR28_EL0}}, x12        // encoding: [0x8c,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmevcntr29_el0|PMEVCNTR29_EL0}}, x12        // encoding: [0xac,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmevcntr30_el0|PMEVCNTR30_EL0}}, x12        // encoding: [0xcc,0xeb,0x1b,0xd5]
// CHECK: msr      {{pmccfiltr_el0|PMCCFILTR_EL0}}, x12         // encoding: [0xec,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper0_el0|PMEVTYPER0_EL0}}, x12        // encoding: [0x0c,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper1_el0|PMEVTYPER1_EL0}}, x12        // encoding: [0x2c,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper2_el0|PMEVTYPER2_EL0}}, x12        // encoding: [0x4c,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper3_el0|PMEVTYPER3_EL0}}, x12        // encoding: [0x6c,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper4_el0|PMEVTYPER4_EL0}}, x12        // encoding: [0x8c,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper5_el0|PMEVTYPER5_EL0}}, x12        // encoding: [0xac,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper6_el0|PMEVTYPER6_EL0}}, x12        // encoding: [0xcc,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper7_el0|PMEVTYPER7_EL0}}, x12        // encoding: [0xec,0xec,0x1b,0xd5]
// CHECK: msr      {{pmevtyper8_el0|PMEVTYPER8_EL0}}, x12        // encoding: [0x0c,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper9_el0|PMEVTYPER9_EL0}}, x12        // encoding: [0x2c,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper10_el0|PMEVTYPER10_EL0}}, x12       // encoding: [0x4c,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper11_el0|PMEVTYPER11_EL0}}, x12       // encoding: [0x6c,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper12_el0|PMEVTYPER12_EL0}}, x12       // encoding: [0x8c,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper13_el0|PMEVTYPER13_EL0}}, x12       // encoding: [0xac,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper14_el0|PMEVTYPER14_EL0}}, x12       // encoding: [0xcc,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper15_el0|PMEVTYPER15_EL0}}, x12       // encoding: [0xec,0xed,0x1b,0xd5]
// CHECK: msr      {{pmevtyper16_el0|PMEVTYPER16_EL0}}, x12       // encoding: [0x0c,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper17_el0|PMEVTYPER17_EL0}}, x12       // encoding: [0x2c,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper18_el0|PMEVTYPER18_EL0}}, x12       // encoding: [0x4c,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper19_el0|PMEVTYPER19_EL0}}, x12       // encoding: [0x6c,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper20_el0|PMEVTYPER20_EL0}}, x12       // encoding: [0x8c,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper21_el0|PMEVTYPER21_EL0}}, x12       // encoding: [0xac,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper22_el0|PMEVTYPER22_EL0}}, x12       // encoding: [0xcc,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper23_el0|PMEVTYPER23_EL0}}, x12       // encoding: [0xec,0xee,0x1b,0xd5]
// CHECK: msr      {{pmevtyper24_el0|PMEVTYPER24_EL0}}, x12       // encoding: [0x0c,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper25_el0|PMEVTYPER25_EL0}}, x12       // encoding: [0x2c,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper26_el0|PMEVTYPER26_EL0}}, x12       // encoding: [0x4c,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper27_el0|PMEVTYPER27_EL0}}, x12       // encoding: [0x6c,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper28_el0|PMEVTYPER28_EL0}}, x12       // encoding: [0x8c,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper29_el0|PMEVTYPER29_EL0}}, x12       // encoding: [0xac,0xef,0x1b,0xd5]
// CHECK: msr      {{pmevtyper30_el0|PMEVTYPER30_EL0}}, x12       // encoding: [0xcc,0xef,0x1b,0xd5]
// CHECK: msr      {{amair2_el1|AMAIR2_EL1}}, x12       // encoding: [0x2c,0xa3,0x18,0xd5]
// CHECK: msr      {{amair2_el12|AMAIR2_EL12}}, x12       // encoding: [0x2c,0xa3,0x1d,0xd5]
// CHECK: msr      {{amair2_el2|AMAIR2_EL2}}, x12       // encoding: [0x2c,0xa3,0x1c,0xd5]
// CHECK: msr      {{amair2_el3|AMAIR2_EL3}}, x12       // encoding: [0x2c,0xa3,0x1e,0xd5]
// CHECK: msr      {{mair2_el1|MAIR2_EL1}}, x12       // encoding: [0x2c,0xa2,0x18,0xd5]
// CHECK: msr      {{mair2_el12|MAIR2_EL12}}, x12       // encoding: [0x2c,0xa2,0x1d,0xd5]
// CHECK: msr      {{mair2_el2|MAIR2_EL2}}, x12       // encoding: [0x2c,0xa1,0x1c,0xd5]
// CHECK: msr      {{mair2_el3|MAIR2_EL3}}, x12       // encoding: [0x2c,0xa1,0x1e,0xd5]
// CHECK: msr      {{pire0_el1|PIRE0_EL1}}, x12       // encoding: [0x4c,0xa2,0x18,0xd5]
// CHECK: msr      {{pire0_el2|PIRE0_EL2}}, x12       // encoding: [0x4c,0xa2,0x1c,0xd5]
// CHECK: msr      {{pir_el1|PIR_EL1}}, x12       // encoding: [0x6c,0xa2,0x18,0xd5]
// CHECK: msr      {{pir_el12|PIR_EL12}}, x12       // encoding: [0x6c,0xa2,0x1d,0xd5]
// CHECK: msr      {{pir_el2|PIR_EL2}}, x12       // encoding: [0x6c,0xa2,0x1c,0xd5]
// CHECK: msr      {{pir_el3|PIR_EL3}}, x12       // encoding: [0x6c,0xa2,0x1e,0xd5]
// CHECK: msr      {{s2pir_el2|S2PIR_EL2}}, x12       // encoding: [0xac,0xa2,0x1c,0xd5]
// CHECK: msr      {{por_el0|POR_EL0}}, x12       // encoding: [0x8c,0xa2,0x1b,0xd5]
// CHECK: msr      {{por_el1|POR_EL1}}, x12       // encoding: [0x8c,0xa2,0x18,0xd5]
// CHECK: msr      {{por_el12|POR_EL12}}, x12       // encoding: [0x8c,0xa2,0x1d,0xd5]
// CHECK: msr      {{por_el2|POR_EL2}}, x12       // encoding: [0x8c,0xa2,0x1c,0xd5]
// CHECK: msr      {{por_el3|POR_EL3}}, x12       // encoding: [0x8c,0xa2,0x1e,0xd5]
// CHECK: msr      {{s2por_el1|S2POR_EL1}}, x12       // encoding: [0xac,0xa2,0x18,0xd5]
// CHECK: msr      {{sctlr2_el1|SCTLR2_EL1}}, x12       // encoding: [0x6c,0x10,0x18,0xd5]
// CHECK: msr      {{sctlr2_el12|SCTLR2_EL12}}, x12       // encoding: [0x6c,0x10,0x1d,0xd5]
// CHECK: msr      {{sctlr2_el2|SCTLR2_EL2}}, x12       // encoding: [0x6c,0x10,0x1c,0xd5]
// CHECK: msr      {{tcr2_el1|TCR2_EL1}}, x12       // encoding: [0x6c,0x20,0x18,0xd5]
// CHECK: msr      {{tcr2_el12|TCR2_EL12}}, x12       // encoding: [0x6c,0x20,0x1d,0xd5]
// CHECK: msr      {{tcr2_el2|TCR2_EL2}}, x12       // encoding: [0x6c,0x20,0x1c,0xd5]

	mrs x9, TEECR32_EL1
	mrs x9, OSDTRRX_EL1
	mrs x9, MDCCSR_EL0
	mrs x9, MDCCINT_EL1
	mrs x9, MDSCR_EL1
	mrs x9, OSDTRTX_EL1
	mrs x9, DBGDTR_EL0
	mrs x9, DBGDTRRX_EL0
	mrs x9, OSECCR_EL1
	mrs x9, DBGVCR32_EL2
	mrs x9, DBGBVR0_EL1
	mrs x9, DBGBVR1_EL1
	mrs x9, DBGBVR2_EL1
	mrs x9, DBGBVR3_EL1
	mrs x9, DBGBVR4_EL1
	mrs x9, DBGBVR5_EL1
	mrs x9, DBGBVR6_EL1
	mrs x9, DBGBVR7_EL1
	mrs x9, DBGBVR8_EL1
	mrs x9, DBGBVR9_EL1
	mrs x9, DBGBVR10_EL1
	mrs x9, DBGBVR11_EL1
	mrs x9, DBGBVR12_EL1
	mrs x9, DBGBVR13_EL1
	mrs x9, DBGBVR14_EL1
	mrs x9, DBGBVR15_EL1
	mrs x9, DBGBCR0_EL1
	mrs x9, DBGBCR1_EL1
	mrs x9, DBGBCR2_EL1
	mrs x9, DBGBCR3_EL1
	mrs x9, DBGBCR4_EL1
	mrs x9, DBGBCR5_EL1
	mrs x9, DBGBCR6_EL1
	mrs x9, DBGBCR7_EL1
	mrs x9, DBGBCR8_EL1
	mrs x9, DBGBCR9_EL1
	mrs x9, DBGBCR10_EL1
	mrs x9, DBGBCR11_EL1
	mrs x9, DBGBCR12_EL1
	mrs x9, DBGBCR13_EL1
	mrs x9, DBGBCR14_EL1
	mrs x9, DBGBCR15_EL1
	mrs x9, DBGWVR0_EL1
	mrs x9, DBGWVR1_EL1
	mrs x9, DBGWVR2_EL1
	mrs x9, DBGWVR3_EL1
	mrs x9, DBGWVR4_EL1
	mrs x9, DBGWVR5_EL1
	mrs x9, DBGWVR6_EL1
	mrs x9, DBGWVR7_EL1
	mrs x9, DBGWVR8_EL1
	mrs x9, DBGWVR9_EL1
	mrs x9, DBGWVR10_EL1
	mrs x9, DBGWVR11_EL1
	mrs x9, DBGWVR12_EL1
	mrs x9, DBGWVR13_EL1
	mrs x9, DBGWVR14_EL1
	mrs x9, DBGWVR15_EL1
	mrs x9, DBGWCR0_EL1
	mrs x9, DBGWCR1_EL1
	mrs x9, DBGWCR2_EL1
	mrs x9, DBGWCR3_EL1
	mrs x9, DBGWCR4_EL1
	mrs x9, DBGWCR5_EL1
	mrs x9, DBGWCR6_EL1
	mrs x9, DBGWCR7_EL1
	mrs x9, DBGWCR8_EL1
	mrs x9, DBGWCR9_EL1
	mrs x9, DBGWCR10_EL1
	mrs x9, DBGWCR11_EL1
	mrs x9, DBGWCR12_EL1
	mrs x9, DBGWCR13_EL1
	mrs x9, DBGWCR14_EL1
	mrs x9, DBGWCR15_EL1
	mrs x9, MDRAR_EL1
	mrs x9, TEEHBR32_EL1
	mrs x9, OSLSR_EL1
	mrs x9, OSDLR_EL1
	mrs x9, DBGPRCR_EL1
	mrs x9, DBGCLAIMSET_EL1
	mrs x9, DBGCLAIMCLR_EL1
	mrs x9, DBGAUTHSTATUS_EL1
	mrs x9, MIDR_EL1
	mrs x9, CCSIDR_EL1
	mrs x9, CSSELR_EL1
	mrs x9, VPIDR_EL2
	mrs x9, CLIDR_EL1
	mrs x9, CTR_EL0
	mrs x9, MPIDR_EL1
	mrs x9, VMPIDR_EL2
	mrs x9, REVIDR_EL1
	mrs x9, AIDR_EL1
	mrs x9, DCZID_EL0
	mrs x9, ID_PFR0_EL1
	mrs x9, ID_PFR1_EL1
	mrs x9, ID_DFR0_EL1
	mrs x9, ID_DFR1_EL1
	mrs x9, ID_AFR0_EL1
	mrs x9, ID_MMFR0_EL1
	mrs x9, ID_MMFR1_EL1
	mrs x9, ID_MMFR2_EL1
	mrs x9, ID_MMFR3_EL1
	mrs x9, ID_MMFR4_EL1
	mrs x9, ID_MMFR5_EL1
	mrs x9, ID_ISAR0_EL1
	mrs x9, ID_ISAR1_EL1
	mrs x9, ID_ISAR2_EL1
	mrs x9, ID_ISAR3_EL1
	mrs x9, ID_ISAR4_EL1
	mrs x9, ID_ISAR5_EL1
	mrs x9, MVFR0_EL1
	mrs x9, MVFR1_EL1
	mrs x9, MVFR2_EL1
	mrs x9, ID_AA64PFR0_EL1
	mrs x9, ID_AA64PFR1_EL1
	mrs x9, ID_AA64DFR0_EL1
	mrs x9, ID_AA64DFR1_EL1
	mrs x9, ID_AA64AFR0_EL1
	mrs x9, ID_AA64AFR1_EL1
	mrs x9, ID_AA64ISAR0_EL1
	mrs x9, ID_AA64ISAR1_EL1
	mrs x9, ID_AA64MMFR0_EL1
	mrs x9, ID_AA64MMFR1_EL1
	mrs x9, SCTLR_EL1
	mrs x9, SCTLR_EL2
	mrs x9, SCTLR_EL3
	mrs x9, ACTLR_EL1
	mrs x9, ACTLR_EL2
	mrs x9, ACTLR_EL3
	mrs x9, CPACR_EL1
	mrs x9, HCR_EL2
	mrs x9, SCR_EL3
	mrs x9, MDCR_EL2
	mrs x9, SDER32_EL3
	mrs x9, CPTR_EL2
	mrs x9, CPTR_EL3
	mrs x9, HSTR_EL2
	mrs x9, HACR_EL2
	mrs x9, MDCR_EL3
	mrs x9, TTBR0_EL1
	mrs x9, TTBR0_EL2
	mrs x9, TTBR0_EL3
	mrs x9, TTBR1_EL1
	mrs x9, TCR_EL1
	mrs x9, TCR_EL2
	mrs x9, TCR_EL3
	mrs x9, VTTBR_EL2
	mrs x9, VTCR_EL2
	mrs x9, DACR32_EL2
	mrs x9, SPSR_EL1
	mrs x9, SPSR_EL2
	mrs x9, SPSR_EL3
	mrs x9, ELR_EL1
	mrs x9, ELR_EL2
	mrs x9, ELR_EL3
	mrs x9, SP_EL0
	mrs x9, SP_EL1
	mrs x9, SP_EL2
	mrs x9, SPSel
	mrs x9, NZCV
	mrs x9, DAIF
	mrs x9, CurrentEL
	mrs x9, SPSR_irq
	mrs x9, SPSR_abt
	mrs x9, SPSR_und
	mrs x9, SPSR_fiq
	mrs x9, FPCR
	mrs x9, FPSR
	mrs x9, DSPSR_EL0
	mrs x9, DLR_EL0
	mrs x9, IFSR32_EL2
	mrs x9, AFSR0_EL1
	mrs x9, AFSR0_EL2
	mrs x9, AFSR0_EL3
	mrs x9, AFSR1_EL1
	mrs x9, AFSR1_EL2
	mrs x9, AFSR1_EL3
	mrs x9, ESR_EL1
	mrs x9, ESR_EL2
	mrs x9, ESR_EL3
	mrs x9, FPEXC32_EL2
	mrs x9, FAR_EL1
	mrs x9, FAR_EL2
	mrs x9, FAR_EL3
	mrs x9, HPFAR_EL2
	mrs x9, PAR_EL1
	mrs x9, PMCR_EL0
	mrs x9, PMCNTENSET_EL0
	mrs x9, PMCNTENCLR_EL0
	mrs x9, PMOVSCLR_EL0
	mrs x9, PMSELR_EL0
	mrs x9, PMCEID0_EL0
	mrs x9, PMCEID1_EL0
	mrs x9, PMCCNTR_EL0
	mrs x9, PMXEVTYPER_EL0
	mrs x9, PMXEVCNTR_EL0
	mrs x9, PMUSERENR_EL0
	mrs x9, PMINTENSET_EL1
	mrs x9, PMINTENCLR_EL1
	mrs x9, PMOVSSET_EL0
	mrs x9, PMMIR_EL1
	mrs x9, MAIR_EL1
	mrs x9, MAIR_EL2
	mrs x9, MAIR_EL3
	mrs x9, AMAIR_EL1
	mrs x9, AMAIR_EL2
	mrs x9, AMAIR_EL3
	mrs x9, VBAR_EL1
	mrs x9, VBAR_EL2
	mrs x9, VBAR_EL3
	mrs x9, RVBAR_EL1
	mrs x9, RVBAR_EL2
	mrs x9, RVBAR_EL3
	mrs x9, RMR_EL1
	mrs x9, RMR_EL2
	mrs x9, RMR_EL3
	mrs x9, ISR_EL1
	mrs x9, CONTEXTIDR_EL1
	mrs x9, TPIDR_EL0
	mrs x9, TPIDR_EL2
	mrs x9, TPIDR_EL3
	mrs x9, TPIDRRO_EL0
	mrs x9, TPIDR_EL1
	mrs x9, CNTFRQ_EL0
	mrs x9, CNTPCT_EL0
	mrs x9, CNTVCT_EL0
	mrs x9, CNTVOFF_EL2
	mrs x9, CNTKCTL_EL1
	mrs x9, CNTHCTL_EL2
	mrs x9, CNTP_TVAL_EL0
	mrs x9, CNTHP_TVAL_EL2
	mrs x9, CNTPS_TVAL_EL1
	mrs x9, CNTP_CTL_EL0
	mrs x9, CNTHP_CTL_EL2
	mrs x9, CNTPS_CTL_EL1
	mrs x9, CNTP_CVAL_EL0
	mrs x9, CNTHP_CVAL_EL2
	mrs x9, CNTPS_CVAL_EL1
	mrs x9, CNTV_TVAL_EL0
	mrs x9, CNTV_CTL_EL0
	mrs x9, CNTV_CVAL_EL0
	mrs x9, PMEVCNTR0_EL0
	mrs x9, PMEVCNTR1_EL0
	mrs x9, PMEVCNTR2_EL0
	mrs x9, PMEVCNTR3_EL0
	mrs x9, PMEVCNTR4_EL0
	mrs x9, PMEVCNTR5_EL0
	mrs x9, PMEVCNTR6_EL0
	mrs x9, PMEVCNTR7_EL0
	mrs x9, PMEVCNTR8_EL0
	mrs x9, PMEVCNTR9_EL0
	mrs x9, PMEVCNTR10_EL0
	mrs x9, PMEVCNTR11_EL0
	mrs x9, PMEVCNTR12_EL0
	mrs x9, PMEVCNTR13_EL0
	mrs x9, PMEVCNTR14_EL0
	mrs x9, PMEVCNTR15_EL0
	mrs x9, PMEVCNTR16_EL0
	mrs x9, PMEVCNTR17_EL0
	mrs x9, PMEVCNTR18_EL0
	mrs x9, PMEVCNTR19_EL0
	mrs x9, PMEVCNTR20_EL0
	mrs x9, PMEVCNTR21_EL0
	mrs x9, PMEVCNTR22_EL0
	mrs x9, PMEVCNTR23_EL0
	mrs x9, PMEVCNTR24_EL0
	mrs x9, PMEVCNTR25_EL0
	mrs x9, PMEVCNTR26_EL0
	mrs x9, PMEVCNTR27_EL0
	mrs x9, PMEVCNTR28_EL0
	mrs x9, PMEVCNTR29_EL0
	mrs x9, PMEVCNTR30_EL0
	mrs x9, PMCCFILTR_EL0
	mrs x9, PMEVTYPER0_EL0
	mrs x9, PMEVTYPER1_EL0
	mrs x9, PMEVTYPER2_EL0
	mrs x9, PMEVTYPER3_EL0
	mrs x9, PMEVTYPER4_EL0
	mrs x9, PMEVTYPER5_EL0
	mrs x9, PMEVTYPER6_EL0
	mrs x9, PMEVTYPER7_EL0
	mrs x9, PMEVTYPER8_EL0
	mrs x9, PMEVTYPER9_EL0
	mrs x9, PMEVTYPER10_EL0
	mrs x9, PMEVTYPER11_EL0
	mrs x9, PMEVTYPER12_EL0
	mrs x9, PMEVTYPER13_EL0
	mrs x9, PMEVTYPER14_EL0
	mrs x9, PMEVTYPER15_EL0
	mrs x9, PMEVTYPER16_EL0
	mrs x9, PMEVTYPER17_EL0
	mrs x9, PMEVTYPER18_EL0
	mrs x9, PMEVTYPER19_EL0
	mrs x9, PMEVTYPER20_EL0
	mrs x9, PMEVTYPER21_EL0
	mrs x9, PMEVTYPER22_EL0
	mrs x9, PMEVTYPER23_EL0
	mrs x9, PMEVTYPER24_EL0
	mrs x9, PMEVTYPER25_EL0
	mrs x9, PMEVTYPER26_EL0
	mrs x9, PMEVTYPER27_EL0
	mrs x9, PMEVTYPER28_EL0
	mrs x9, PMEVTYPER29_EL0
	mrs x9, PMEVTYPER30_EL0
        mrs x9, AMAIR2_EL1
        mrs x9, AMAIR2_EL12
        mrs x9, AMAIR2_EL2
        mrs x9, AMAIR2_EL3
        mrs x9, MAIR2_EL1
        mrs x9, MAIR2_EL12
        mrs x9, MAIR2_EL2
        mrs x9, MAIR2_EL3
        mrs x9, PIRE0_EL1
        mrs x9, PIRE0_EL2
        mrs x9, PIR_EL1
        mrs x9, PIR_EL12
        mrs x9, PIR_EL2
        mrs x9, PIR_EL3
        mrs x9, S2PIR_EL2
        mrs x9, POR_EL0
        mrs x9, POR_EL1
        mrs x9, POR_EL12
        mrs x9, POR_EL2
        mrs x9, POR_EL3
        mrs x9, S2POR_EL1
        mrs x9, SCTLR2_EL1
        mrs x9, SCTLR2_EL12
        mrs x9, SCTLR2_EL2
        mrs x9, TCR2_EL1
        mrs x9, TCR2_EL12
        mrs x9, TCR2_EL2
// CHECK: mrs      x9, {{teecr32_el1|TEECR32_EL1}}            // encoding: [0x09,0x00,0x32,0xd5]
// CHECK: mrs      x9, {{osdtrrx_el1|OSDTRRX_EL1}}            // encoding: [0x49,0x00,0x30,0xd5]
// CHECK: mrs      x9, {{mdccsr_el0|MDCCSR_EL0}}             // encoding: [0x09,0x01,0x33,0xd5]
// CHECK: mrs      x9, {{mdccint_el1|MDCCINT_EL1}}            // encoding: [0x09,0x02,0x30,0xd5]
// CHECK: mrs      x9, {{mdscr_el1|MDSCR_EL1}}              // encoding: [0x49,0x02,0x30,0xd5]
// CHECK: mrs      x9, {{osdtrtx_el1|OSDTRTX_EL1}}            // encoding: [0x49,0x03,0x30,0xd5]
// CHECK: mrs      x9, {{dbgdtr_el0|DBGDTR_EL0}}             // encoding: [0x09,0x04,0x33,0xd5]
// CHECK: mrs      x9, {{dbgdtrrx_el0|DBGDTRRX_EL0}}           // encoding: [0x09,0x05,0x33,0xd5]
// CHECK: mrs      x9, {{oseccr_el1|OSECCR_EL1}}             // encoding: [0x49,0x06,0x30,0xd5]
// CHECK: mrs      x9, {{dbgvcr32_el2|DBGVCR32_EL2}}           // encoding: [0x09,0x07,0x34,0xd5]
// CHECK: mrs      x9, {{dbgbvr0_el1|DBGBVR0_EL1}}            // encoding: [0x89,0x00,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr1_el1|DBGBVR1_EL1}}            // encoding: [0x89,0x01,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr2_el1|DBGBVR2_EL1}}            // encoding: [0x89,0x02,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr3_el1|DBGBVR3_EL1}}            // encoding: [0x89,0x03,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr4_el1|DBGBVR4_EL1}}            // encoding: [0x89,0x04,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr5_el1|DBGBVR5_EL1}}            // encoding: [0x89,0x05,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr6_el1|DBGBVR6_EL1}}            // encoding: [0x89,0x06,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr7_el1|DBGBVR7_EL1}}            // encoding: [0x89,0x07,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr8_el1|DBGBVR8_EL1}}            // encoding: [0x89,0x08,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr9_el1|DBGBVR9_EL1}}            // encoding: [0x89,0x09,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr10_el1|DBGBVR10_EL1}}           // encoding: [0x89,0x0a,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr11_el1|DBGBVR11_EL1}}           // encoding: [0x89,0x0b,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr12_el1|DBGBVR12_EL1}}           // encoding: [0x89,0x0c,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr13_el1|DBGBVR13_EL1}}           // encoding: [0x89,0x0d,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr14_el1|DBGBVR14_EL1}}           // encoding: [0x89,0x0e,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbvr15_el1|DBGBVR15_EL1}}           // encoding: [0x89,0x0f,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr0_el1|DBGBCR0_EL1}}            // encoding: [0xa9,0x00,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr1_el1|DBGBCR1_EL1}}            // encoding: [0xa9,0x01,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr2_el1|DBGBCR2_EL1}}            // encoding: [0xa9,0x02,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr3_el1|DBGBCR3_EL1}}            // encoding: [0xa9,0x03,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr4_el1|DBGBCR4_EL1}}            // encoding: [0xa9,0x04,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr5_el1|DBGBCR5_EL1}}            // encoding: [0xa9,0x05,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr6_el1|DBGBCR6_EL1}}            // encoding: [0xa9,0x06,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr7_el1|DBGBCR7_EL1}}            // encoding: [0xa9,0x07,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr8_el1|DBGBCR8_EL1}}            // encoding: [0xa9,0x08,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr9_el1|DBGBCR9_EL1}}            // encoding: [0xa9,0x09,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr10_el1|DBGBCR10_EL1}}           // encoding: [0xa9,0x0a,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr11_el1|DBGBCR11_EL1}}           // encoding: [0xa9,0x0b,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr12_el1|DBGBCR12_EL1}}           // encoding: [0xa9,0x0c,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr13_el1|DBGBCR13_EL1}}           // encoding: [0xa9,0x0d,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr14_el1|DBGBCR14_EL1}}           // encoding: [0xa9,0x0e,0x30,0xd5]
// CHECK: mrs      x9, {{dbgbcr15_el1|DBGBCR15_EL1}}           // encoding: [0xa9,0x0f,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr0_el1|DBGWVR0_EL1}}            // encoding: [0xc9,0x00,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr1_el1|DBGWVR1_EL1}}            // encoding: [0xc9,0x01,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr2_el1|DBGWVR2_EL1}}            // encoding: [0xc9,0x02,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr3_el1|DBGWVR3_EL1}}            // encoding: [0xc9,0x03,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr4_el1|DBGWVR4_EL1}}            // encoding: [0xc9,0x04,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr5_el1|DBGWVR5_EL1}}            // encoding: [0xc9,0x05,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr6_el1|DBGWVR6_EL1}}            // encoding: [0xc9,0x06,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr7_el1|DBGWVR7_EL1}}            // encoding: [0xc9,0x07,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr8_el1|DBGWVR8_EL1}}            // encoding: [0xc9,0x08,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr9_el1|DBGWVR9_EL1}}            // encoding: [0xc9,0x09,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr10_el1|DBGWVR10_EL1}}           // encoding: [0xc9,0x0a,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr11_el1|DBGWVR11_EL1}}           // encoding: [0xc9,0x0b,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr12_el1|DBGWVR12_EL1}}           // encoding: [0xc9,0x0c,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr13_el1|DBGWVR13_EL1}}           // encoding: [0xc9,0x0d,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr14_el1|DBGWVR14_EL1}}           // encoding: [0xc9,0x0e,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwvr15_el1|DBGWVR15_EL1}}           // encoding: [0xc9,0x0f,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr0_el1|DBGWCR0_EL1}}            // encoding: [0xe9,0x00,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr1_el1|DBGWCR1_EL1}}            // encoding: [0xe9,0x01,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr2_el1|DBGWCR2_EL1}}            // encoding: [0xe9,0x02,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr3_el1|DBGWCR3_EL1}}            // encoding: [0xe9,0x03,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr4_el1|DBGWCR4_EL1}}            // encoding: [0xe9,0x04,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr5_el1|DBGWCR5_EL1}}            // encoding: [0xe9,0x05,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr6_el1|DBGWCR6_EL1}}            // encoding: [0xe9,0x06,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr7_el1|DBGWCR7_EL1}}            // encoding: [0xe9,0x07,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr8_el1|DBGWCR8_EL1}}            // encoding: [0xe9,0x08,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr9_el1|DBGWCR9_EL1}}            // encoding: [0xe9,0x09,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr10_el1|DBGWCR10_EL1}}           // encoding: [0xe9,0x0a,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr11_el1|DBGWCR11_EL1}}           // encoding: [0xe9,0x0b,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr12_el1|DBGWCR12_EL1}}           // encoding: [0xe9,0x0c,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr13_el1|DBGWCR13_EL1}}           // encoding: [0xe9,0x0d,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr14_el1|DBGWCR14_EL1}}           // encoding: [0xe9,0x0e,0x30,0xd5]
// CHECK: mrs      x9, {{dbgwcr15_el1|DBGWCR15_EL1}}           // encoding: [0xe9,0x0f,0x30,0xd5]
// CHECK: mrs      x9, {{mdrar_el1|MDRAR_EL1}}              // encoding: [0x09,0x10,0x30,0xd5]
// CHECK: mrs      x9, {{teehbr32_el1|TEEHBR32_EL1}}           // encoding: [0x09,0x10,0x32,0xd5]
// CHECK: mrs      x9, {{oslsr_el1|OSLSR_EL1}}              // encoding: [0x89,0x11,0x30,0xd5]
// CHECK: mrs      x9, {{osdlr_el1|OSDLR_EL1}}              // encoding: [0x89,0x13,0x30,0xd5]
// CHECK: mrs      x9, {{dbgprcr_el1|DBGPRCR_EL1}}            // encoding: [0x89,0x14,0x30,0xd5]
// CHECK: mrs      x9, {{dbgclaimset_el1|DBGCLAIMSET_EL1}}        // encoding: [0xc9,0x78,0x30,0xd5]
// CHECK: mrs      x9, {{dbgclaimclr_el1|DBGCLAIMCLR_EL1}}        // encoding: [0xc9,0x79,0x30,0xd5]
// CHECK: mrs      x9, {{dbgauthstatus_el1|DBGAUTHSTATUS_EL1}}      // encoding: [0xc9,0x7e,0x30,0xd5]
// CHECK: mrs      x9, {{midr_el1|MIDR_EL1}}               // encoding: [0x09,0x00,0x38,0xd5]
// CHECK: mrs      x9, {{ccsidr_el1|CCSIDR_EL1}}             // encoding: [0x09,0x00,0x39,0xd5]
// CHECK: mrs      x9, {{csselr_el1|CSSELR_EL1}}             // encoding: [0x09,0x00,0x3a,0xd5]
// CHECK: mrs      x9, {{vpidr_el2|VPIDR_EL2}}              // encoding: [0x09,0x00,0x3c,0xd5]
// CHECK: mrs      x9, {{clidr_el1|CLIDR_EL1}}              // encoding: [0x29,0x00,0x39,0xd5]
// CHECK: mrs      x9, {{ctr_el0|CTR_EL0}}                // encoding: [0x29,0x00,0x3b,0xd5]
// CHECK: mrs      x9, {{mpidr_el1|MPIDR_EL1}}              // encoding: [0xa9,0x00,0x38,0xd5]
// CHECK: mrs      x9, {{vmpidr_el2|VMPIDR_EL2}}             // encoding: [0xa9,0x00,0x3c,0xd5]
// CHECK: mrs      x9, {{revidr_el1|REVIDR_EL1}}             // encoding: [0xc9,0x00,0x38,0xd5]
// CHECK: mrs      x9, {{aidr_el1|AIDR_EL1}}               // encoding: [0xe9,0x00,0x39,0xd5]
// CHECK: mrs      x9, {{dczid_el0|DCZID_EL0}}              // encoding: [0xe9,0x00,0x3b,0xd5]
// CHECK: mrs      x9, {{id_pfr0_el1|ID_PFR0_EL1}}            // encoding: [0x09,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_pfr1_el1|ID_PFR1_EL1}}            // encoding: [0x29,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_dfr0_el1|ID_DFR0_EL1}}            // encoding: [0x49,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_dfr1_el1|ID_DFR1_EL1}}            // encoding: [0xa9,0x03,0x38,0xd5]
// CHECK: mrs      x9, {{id_afr0_el1|ID_AFR0_EL1}}            // encoding: [0x69,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_mmfr0_el1|ID_MMFR0_EL1}}           // encoding: [0x89,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_mmfr1_el1|ID_MMFR1_EL1}}           // encoding: [0xa9,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_mmfr2_el1|ID_MMFR2_EL1}}           // encoding: [0xc9,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_mmfr3_el1|ID_MMFR3_EL1}}           // encoding: [0xe9,0x01,0x38,0xd5]
// CHECK: mrs      x9, {{id_mmfr4_el1|ID_MMFR4_EL1}}           // encoding: [0xc9,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{id_mmfr5_el1|ID_MMFR5_EL1}}           // encoding: [0xc9,0x03,0x38,0xd5]
// CHECK: mrs      x9, {{id_isar0_el1|ID_ISAR0_EL1}}           // encoding: [0x09,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{id_isar1_el1|ID_ISAR1_EL1}}           // encoding: [0x29,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{id_isar2_el1|ID_ISAR2_EL1}}           // encoding: [0x49,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{id_isar3_el1|ID_ISAR3_EL1}}           // encoding: [0x69,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{id_isar4_el1|ID_ISAR4_EL1}}           // encoding: [0x89,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{id_isar5_el1|ID_ISAR5_EL1}}           // encoding: [0xa9,0x02,0x38,0xd5]
// CHECK: mrs      x9, {{mvfr0_el1|MVFR0_EL1}}              // encoding: [0x09,0x03,0x38,0xd5]
// CHECK: mrs      x9, {{mvfr1_el1|MVFR1_EL1}}              // encoding: [0x29,0x03,0x38,0xd5]
// CHECK: mrs      x9, {{mvfr2_el1|MVFR2_EL1}}              // encoding: [0x49,0x03,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64pfr0_el1|ID_AA64PFR0_EL1}}        // encoding: [0x09,0x04,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64pfr1_el1|ID_AA64PFR1_EL1}}        // encoding: [0x29,0x04,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64dfr0_el1|ID_AA64DFR0_EL1}}        // encoding: [0x09,0x05,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64dfr1_el1|ID_AA64DFR1_EL1}}        // encoding: [0x29,0x05,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64afr0_el1|ID_AA64AFR0_EL1}}        // encoding: [0x89,0x05,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64afr1_el1|ID_AA64AFR1_EL1}}        // encoding: [0xa9,0x05,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64isar0_el1|ID_AA64ISAR0_EL1}}       // encoding: [0x09,0x06,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64isar1_el1|ID_AA64ISAR1_EL1}}       // encoding: [0x29,0x06,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64mmfr0_el1|ID_AA64MMFR0_EL1}}       // encoding: [0x09,0x07,0x38,0xd5]
// CHECK: mrs      x9, {{id_aa64mmfr1_el1|ID_AA64MMFR1_EL1}}       // encoding: [0x29,0x07,0x38,0xd5]
// CHECK: mrs      x9, {{sctlr_el1|SCTLR_EL1}}              // encoding: [0x09,0x10,0x38,0xd5]
// CHECK: mrs      x9, {{sctlr_el2|SCTLR_EL2}}              // encoding: [0x09,0x10,0x3c,0xd5]
// CHECK: mrs      x9, {{sctlr_el3|SCTLR_EL3}}              // encoding: [0x09,0x10,0x3e,0xd5]
// CHECK: mrs      x9, {{actlr_el1|ACTLR_EL1}}              // encoding: [0x29,0x10,0x38,0xd5]
// CHECK: mrs      x9, {{actlr_el2|ACTLR_EL2}}              // encoding: [0x29,0x10,0x3c,0xd5]
// CHECK: mrs      x9, {{actlr_el3|ACTLR_EL3}}              // encoding: [0x29,0x10,0x3e,0xd5]
// CHECK: mrs      x9, {{cpacr_el1|CPACR_EL1}}              // encoding: [0x49,0x10,0x38,0xd5]
// CHECK: mrs      x9, {{hcr_el2|HCR_EL2}}                // encoding: [0x09,0x11,0x3c,0xd5]
// CHECK: mrs      x9, {{scr_el3|SCR_EL3}}                // encoding: [0x09,0x11,0x3e,0xd5]
// CHECK: mrs      x9, {{mdcr_el2|MDCR_EL2}}               // encoding: [0x29,0x11,0x3c,0xd5]
// CHECK: mrs      x9, {{sder32_el3|SDER32_EL3}}             // encoding: [0x29,0x11,0x3e,0xd5]
// CHECK: mrs      x9, {{cptr_el2|CPTR_EL2}}               // encoding: [0x49,0x11,0x3c,0xd5]
// CHECK: mrs      x9, {{cptr_el3|CPTR_EL3}}               // encoding: [0x49,0x11,0x3e,0xd5]
// CHECK: mrs      x9, {{hstr_el2|HSTR_EL2}}               // encoding: [0x69,0x11,0x3c,0xd5]
// CHECK: mrs      x9, {{hacr_el2|HACR_EL2}}               // encoding: [0xe9,0x11,0x3c,0xd5]
// CHECK: mrs      x9, {{mdcr_el3|MDCR_EL3}}               // encoding: [0x29,0x13,0x3e,0xd5]
// CHECK: mrs      x9, {{ttbr0_el1|TTBR0_EL1}}              // encoding: [0x09,0x20,0x38,0xd5]
// CHECK: mrs      x9, {{ttbr0_el2|TTBR0_EL2}}              // encoding: [0x09,0x20,0x3c,0xd5]
// CHECK: mrs      x9, {{ttbr0_el3|TTBR0_EL3}}              // encoding: [0x09,0x20,0x3e,0xd5]
// CHECK: mrs      x9, {{ttbr1_el1|TTBR1_EL1}}              // encoding: [0x29,0x20,0x38,0xd5]
// CHECK: mrs      x9, {{tcr_el1|TCR_EL1}}                // encoding: [0x49,0x20,0x38,0xd5]
// CHECK: mrs      x9, {{tcr_el2|TCR_EL2}}                // encoding: [0x49,0x20,0x3c,0xd5]
// CHECK: mrs      x9, {{tcr_el3|TCR_EL3}}                // encoding: [0x49,0x20,0x3e,0xd5]
// CHECK: mrs      x9, {{vttbr_el2|VTTBR_EL2}}              // encoding: [0x09,0x21,0x3c,0xd5]
// CHECK: mrs      x9, {{vtcr_el2|VTCR_EL2}}               // encoding: [0x49,0x21,0x3c,0xd5]
// CHECK: mrs      x9, {{dacr32_el2|DACR32_EL2}}             // encoding: [0x09,0x30,0x3c,0xd5]
// CHECK: mrs      x9, {{spsr_el1|SPSR_EL1}}               // encoding: [0x09,0x40,0x38,0xd5]
// CHECK: mrs      x9, {{spsr_el2|SPSR_EL2}}               // encoding: [0x09,0x40,0x3c,0xd5]
// CHECK: mrs      x9, {{spsr_el3|SPSR_EL3}}               // encoding: [0x09,0x40,0x3e,0xd5]
// CHECK: mrs      x9, {{elr_el1|ELR_EL1}}                // encoding: [0x29,0x40,0x38,0xd5]
// CHECK: mrs      x9, {{elr_el2|ELR_EL2}}                // encoding: [0x29,0x40,0x3c,0xd5]
// CHECK: mrs      x9, {{elr_el3|ELR_EL3}}                // encoding: [0x29,0x40,0x3e,0xd5]
// CHECK: mrs      x9, {{sp_el0|SP_EL0}}                 // encoding: [0x09,0x41,0x38,0xd5]
// CHECK: mrs      x9, {{sp_el1|SP_EL1}}                 // encoding: [0x09,0x41,0x3c,0xd5]
// CHECK: mrs      x9, {{sp_el2|SP_EL2}}                 // encoding: [0x09,0x41,0x3e,0xd5]
// CHECK: mrs      x9, {{SPSel|SPSEL}}                  // encoding: [0x09,0x42,0x38,0xd5]
// CHECK: mrs      x9, {{nzcv|NZCV}}                   // encoding: [0x09,0x42,0x3b,0xd5]
// CHECK: mrs      x9, {{daif|DAIF}}                   // encoding: [0x29,0x42,0x3b,0xd5]
// CHECK: mrs      x9, {{CurrentEL|CURRENTEL}}              // encoding: [0x49,0x42,0x38,0xd5]
// CHECK: mrs      x9, {{SPSR_irq|SPSR_IRQ}}               // encoding: [0x09,0x43,0x3c,0xd5]
// CHECK: mrs      x9, {{SPSR_abt|SPSR_ABT}}               // encoding: [0x29,0x43,0x3c,0xd5]
// CHECK: mrs      x9, {{SPSR_und|SPSR_UND}}               // encoding: [0x49,0x43,0x3c,0xd5]
// CHECK: mrs      x9, {{SPSR_fiq|SPSR_FIQ}}               // encoding: [0x69,0x43,0x3c,0xd5]
// CHECK: mrs      x9, {{fpcr|FPCR}}                   // encoding: [0x09,0x44,0x3b,0xd5]
// CHECK: mrs      x9, {{fpsr|FPSR}}                   // encoding: [0x29,0x44,0x3b,0xd5]
// CHECK: mrs      x9, {{dspsr_el0|DSPSR_EL0}}              // encoding: [0x09,0x45,0x3b,0xd5]
// CHECK: mrs      x9, {{dlr_el0|DLR_EL0}}                // encoding: [0x29,0x45,0x3b,0xd5]
// CHECK: mrs      x9, {{ifsr32_el2|IFSR32_EL2}}             // encoding: [0x29,0x50,0x3c,0xd5]
// CHECK: mrs      x9, {{afsr0_el1|AFSR0_EL1}}              // encoding: [0x09,0x51,0x38,0xd5]
// CHECK: mrs      x9, {{afsr0_el2|AFSR0_EL2}}              // encoding: [0x09,0x51,0x3c,0xd5]
// CHECK: mrs      x9, {{afsr0_el3|AFSR0_EL3}}              // encoding: [0x09,0x51,0x3e,0xd5]
// CHECK: mrs      x9, {{afsr1_el1|AFSR1_EL1}}              // encoding: [0x29,0x51,0x38,0xd5]
// CHECK: mrs      x9, {{afsr1_el2|AFSR1_EL2}}              // encoding: [0x29,0x51,0x3c,0xd5]
// CHECK: mrs      x9, {{afsr1_el3|AFSR1_EL3}}              // encoding: [0x29,0x51,0x3e,0xd5]
// CHECK: mrs      x9, {{esr_el1|ESR_EL1}}                // encoding: [0x09,0x52,0x38,0xd5]
// CHECK: mrs      x9, {{esr_el2|ESR_EL2}}                // encoding: [0x09,0x52,0x3c,0xd5]
// CHECK: mrs      x9, {{esr_el3|ESR_EL3}}                // encoding: [0x09,0x52,0x3e,0xd5]
// CHECK: mrs      x9, {{fpexc32_el2|FPEXC32_EL2}}            // encoding: [0x09,0x53,0x3c,0xd5]
// CHECK: mrs      x9, {{far_el1|FAR_EL1}}                // encoding: [0x09,0x60,0x38,0xd5]
// CHECK: mrs      x9, {{far_el2|FAR_EL2}}                // encoding: [0x09,0x60,0x3c,0xd5]
// CHECK: mrs      x9, {{far_el3|FAR_EL3}}                // encoding: [0x09,0x60,0x3e,0xd5]
// CHECK: mrs      x9, {{hpfar_el2|HPFAR_EL2}}              // encoding: [0x89,0x60,0x3c,0xd5]
// CHECK: mrs      x9, {{par_el1|PAR_EL1}}                // encoding: [0x09,0x74,0x38,0xd5]
// CHECK: mrs      x9, {{pmcr_el0|PMCR_EL0}}               // encoding: [0x09,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmcntenset_el0|PMCNTENSET_EL0}}         // encoding: [0x29,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmcntenclr_el0|PMCNTENCLR_EL0}}         // encoding: [0x49,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmovsclr_el0|PMOVSCLR_EL0}}           // encoding: [0x69,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmselr_el0|PMSELR_EL0}}             // encoding: [0xa9,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmceid0_el0|PMCEID0_EL0}}            // encoding: [0xc9,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmceid1_el0|PMCEID1_EL0}}            // encoding: [0xe9,0x9c,0x3b,0xd5]
// CHECK: mrs      x9, {{pmccntr_el0|PMCCNTR_EL0}}            // encoding: [0x09,0x9d,0x3b,0xd5]
// CHECK: mrs      x9, {{pmxevtyper_el0|PMXEVTYPER_EL0}}         // encoding: [0x29,0x9d,0x3b,0xd5]
// CHECK: mrs      x9, {{pmxevcntr_el0|PMXEVCNTR_EL0}}          // encoding: [0x49,0x9d,0x3b,0xd5]
// CHECK: mrs      x9, {{pmuserenr_el0|PMUSERENR_EL0}}          // encoding: [0x09,0x9e,0x3b,0xd5]
// CHECK: mrs      x9, {{pmintenset_el1|PMINTENSET_EL1}}         // encoding: [0x29,0x9e,0x38,0xd5]
// CHECK: mrs      x9, {{pmintenclr_el1|PMINTENCLR_EL1}}         // encoding: [0x49,0x9e,0x38,0xd5]
// CHECK: mrs      x9, {{pmovsset_el0|PMOVSSET_EL0}}           // encoding: [0x69,0x9e,0x3b,0xd5]
// CHECK: mrs      x9, {{pmmir_el1|PMMIR_EL1}}             // encoding: [0xc9,0x9e,0x38,0xd5]
// CHECK: mrs      x9, {{mair_el1|MAIR_EL1}}               // encoding: [0x09,0xa2,0x38,0xd5]
// CHECK: mrs      x9, {{mair_el2|MAIR_EL2}}               // encoding: [0x09,0xa2,0x3c,0xd5]
// CHECK: mrs      x9, {{mair_el3|MAIR_EL3}}               // encoding: [0x09,0xa2,0x3e,0xd5]
// CHECK: mrs      x9, {{amair_el1|AMAIR_EL1}}              // encoding: [0x09,0xa3,0x38,0xd5]
// CHECK: mrs      x9, {{amair_el2|AMAIR_EL2}}              // encoding: [0x09,0xa3,0x3c,0xd5]
// CHECK: mrs      x9, {{amair_el3|AMAIR_EL3}}              // encoding: [0x09,0xa3,0x3e,0xd5]
// CHECK: mrs      x9, {{vbar_el1|VBAR_EL1}}               // encoding: [0x09,0xc0,0x38,0xd5]
// CHECK: mrs      x9, {{vbar_el2|VBAR_EL2}}               // encoding: [0x09,0xc0,0x3c,0xd5]
// CHECK: mrs      x9, {{vbar_el3|VBAR_EL3}}               // encoding: [0x09,0xc0,0x3e,0xd5]
// CHECK: mrs      x9, {{rvbar_el1|RVBAR_EL1}}              // encoding: [0x29,0xc0,0x38,0xd5]
// CHECK: mrs      x9, {{rvbar_el2|RVBAR_EL2}}              // encoding: [0x29,0xc0,0x3c,0xd5]
// CHECK: mrs      x9, {{rvbar_el3|RVBAR_EL3}}              // encoding: [0x29,0xc0,0x3e,0xd5]
// CHECK: mrs      x9, {{rmr_el1|RMR_EL1}}                // encoding: [0x49,0xc0,0x38,0xd5]
// CHECK: mrs      x9, {{rmr_el2|RMR_EL2}}                // encoding: [0x49,0xc0,0x3c,0xd5]
// CHECK: mrs      x9, {{rmr_el3|RMR_EL3}}                // encoding: [0x49,0xc0,0x3e,0xd5]
// CHECK: mrs      x9, {{isr_el1|ISR_EL1}}                // encoding: [0x09,0xc1,0x38,0xd5]
// CHECK: mrs      x9, {{contextidr_el1|CONTEXTIDR_EL1}}         // encoding: [0x29,0xd0,0x38,0xd5]
// CHECK: mrs      x9, {{tpidr_el0|TPIDR_EL0}}              // encoding: [0x49,0xd0,0x3b,0xd5]
// CHECK: mrs      x9, {{tpidr_el2|TPIDR_EL2}}              // encoding: [0x49,0xd0,0x3c,0xd5]
// CHECK: mrs      x9, {{tpidr_el3|TPIDR_EL3}}              // encoding: [0x49,0xd0,0x3e,0xd5]
// CHECK: mrs      x9, {{tpidrro_el0|TPIDRRO_EL0}}            // encoding: [0x69,0xd0,0x3b,0xd5]
// CHECK: mrs      x9, {{tpidr_el1|TPIDR_EL1}}              // encoding: [0x89,0xd0,0x38,0xd5]
// CHECK: mrs      x9, {{cntfrq_el0|CNTFRQ_EL0}}             // encoding: [0x09,0xe0,0x3b,0xd5]
// CHECK: mrs      x9, {{cntpct_el0|CNTPCT_EL0}}             // encoding: [0x29,0xe0,0x3b,0xd5]
// CHECK: mrs      x9, {{cntvct_el0|CNTVCT_EL0}}             // encoding: [0x49,0xe0,0x3b,0xd5]
// CHECK: mrs      x9, {{cntvoff_el2|CNTVOFF_EL2}}            // encoding: [0x69,0xe0,0x3c,0xd5]
// CHECK: mrs      x9, {{cntkctl_el1|CNTKCTL_EL1}}            // encoding: [0x09,0xe1,0x38,0xd5]
// CHECK: mrs      x9, {{cnthctl_el2|CNTHCTL_EL2}}            // encoding: [0x09,0xe1,0x3c,0xd5]
// CHECK: mrs      x9, {{cntp_tval_el0|CNTP_TVAL_EL0}}          // encoding: [0x09,0xe2,0x3b,0xd5]
// CHECK: mrs      x9, {{cnthp_tval_el2|CNTHP_TVAL_EL2}}         // encoding: [0x09,0xe2,0x3c,0xd5]
// CHECK: mrs      x9, {{cntps_tval_el1|CNTPS_TVAL_EL1}}         // encoding: [0x09,0xe2,0x3f,0xd5]
// CHECK: mrs      x9, {{cntp_ctl_el0|CNTP_CTL_EL0}}           // encoding: [0x29,0xe2,0x3b,0xd5]
// CHECK: mrs      x9, {{cnthp_ctl_el2|CNTHP_CTL_EL2}}          // encoding: [0x29,0xe2,0x3c,0xd5]
// CHECK: mrs      x9, {{cntps_ctl_el1|CNTPS_CTL_EL1}}          // encoding: [0x29,0xe2,0x3f,0xd5]
// CHECK: mrs      x9, {{cntp_cval_el0|CNTP_CVAL_EL0}}          // encoding: [0x49,0xe2,0x3b,0xd5]
// CHECK: mrs      x9, {{cnthp_cval_el2|CNTHP_CVAL_EL2}}         // encoding: [0x49,0xe2,0x3c,0xd5]
// CHECK: mrs      x9, {{cntps_cval_el1|CNTPS_CVAL_EL1}}         // encoding: [0x49,0xe2,0x3f,0xd5]
// CHECK: mrs      x9, {{cntv_tval_el0|CNTV_TVAL_EL0}}          // encoding: [0x09,0xe3,0x3b,0xd5]
// CHECK: mrs      x9, {{cntv_ctl_el0|CNTV_CTL_EL0}}           // encoding: [0x29,0xe3,0x3b,0xd5]
// CHECK: mrs      x9, {{cntv_cval_el0|CNTV_CVAL_EL0}}          // encoding: [0x49,0xe3,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr0_el0|PMEVCNTR0_EL0}}          // encoding: [0x09,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr1_el0|PMEVCNTR1_EL0}}          // encoding: [0x29,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr2_el0|PMEVCNTR2_EL0}}          // encoding: [0x49,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr3_el0|PMEVCNTR3_EL0}}          // encoding: [0x69,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr4_el0|PMEVCNTR4_EL0}}          // encoding: [0x89,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr5_el0|PMEVCNTR5_EL0}}          // encoding: [0xa9,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr6_el0|PMEVCNTR6_EL0}}          // encoding: [0xc9,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr7_el0|PMEVCNTR7_EL0}}          // encoding: [0xe9,0xe8,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr8_el0|PMEVCNTR8_EL0}}          // encoding: [0x09,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr9_el0|PMEVCNTR9_EL0}}          // encoding: [0x29,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr10_el0|PMEVCNTR10_EL0}}         // encoding: [0x49,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr11_el0|PMEVCNTR11_EL0}}         // encoding: [0x69,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr12_el0|PMEVCNTR12_EL0}}         // encoding: [0x89,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr13_el0|PMEVCNTR13_EL0}}         // encoding: [0xa9,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr14_el0|PMEVCNTR14_EL0}}         // encoding: [0xc9,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr15_el0|PMEVCNTR15_EL0}}         // encoding: [0xe9,0xe9,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr16_el0|PMEVCNTR16_EL0}}         // encoding: [0x09,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr17_el0|PMEVCNTR17_EL0}}         // encoding: [0x29,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr18_el0|PMEVCNTR18_EL0}}         // encoding: [0x49,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr19_el0|PMEVCNTR19_EL0}}         // encoding: [0x69,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr20_el0|PMEVCNTR20_EL0}}         // encoding: [0x89,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr21_el0|PMEVCNTR21_EL0}}         // encoding: [0xa9,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr22_el0|PMEVCNTR22_EL0}}         // encoding: [0xc9,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr23_el0|PMEVCNTR23_EL0}}         // encoding: [0xe9,0xea,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr24_el0|PMEVCNTR24_EL0}}         // encoding: [0x09,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr25_el0|PMEVCNTR25_EL0}}         // encoding: [0x29,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr26_el0|PMEVCNTR26_EL0}}         // encoding: [0x49,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr27_el0|PMEVCNTR27_EL0}}         // encoding: [0x69,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr28_el0|PMEVCNTR28_EL0}}         // encoding: [0x89,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr29_el0|PMEVCNTR29_EL0}}         // encoding: [0xa9,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevcntr30_el0|PMEVCNTR30_EL0}}         // encoding: [0xc9,0xeb,0x3b,0xd5]
// CHECK: mrs      x9, {{pmccfiltr_el0|PMCCFILTR_EL0}}          // encoding: [0xe9,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper0_el0|PMEVTYPER0_EL0}}         // encoding: [0x09,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper1_el0|PMEVTYPER1_EL0}}         // encoding: [0x29,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper2_el0|PMEVTYPER2_EL0}}         // encoding: [0x49,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper3_el0|PMEVTYPER3_EL0}}         // encoding: [0x69,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper4_el0|PMEVTYPER4_EL0}}         // encoding: [0x89,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper5_el0|PMEVTYPER5_EL0}}         // encoding: [0xa9,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper6_el0|PMEVTYPER6_EL0}}         // encoding: [0xc9,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper7_el0|PMEVTYPER7_EL0}}         // encoding: [0xe9,0xec,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper8_el0|PMEVTYPER8_EL0}}         // encoding: [0x09,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper9_el0|PMEVTYPER9_EL0}}         // encoding: [0x29,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper10_el0|PMEVTYPER10_EL0}}        // encoding: [0x49,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper11_el0|PMEVTYPER11_EL0}}        // encoding: [0x69,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper12_el0|PMEVTYPER12_EL0}}        // encoding: [0x89,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper13_el0|PMEVTYPER13_EL0}}        // encoding: [0xa9,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper14_el0|PMEVTYPER14_EL0}}        // encoding: [0xc9,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper15_el0|PMEVTYPER15_EL0}}        // encoding: [0xe9,0xed,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper16_el0|PMEVTYPER16_EL0}}        // encoding: [0x09,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper17_el0|PMEVTYPER17_EL0}}        // encoding: [0x29,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper18_el0|PMEVTYPER18_EL0}}        // encoding: [0x49,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper19_el0|PMEVTYPER19_EL0}}        // encoding: [0x69,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper20_el0|PMEVTYPER20_EL0}}        // encoding: [0x89,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper21_el0|PMEVTYPER21_EL0}}        // encoding: [0xa9,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper22_el0|PMEVTYPER22_EL0}}        // encoding: [0xc9,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper23_el0|PMEVTYPER23_EL0}}        // encoding: [0xe9,0xee,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper24_el0|PMEVTYPER24_EL0}}        // encoding: [0x09,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper25_el0|PMEVTYPER25_EL0}}        // encoding: [0x29,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper26_el0|PMEVTYPER26_EL0}}        // encoding: [0x49,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper27_el0|PMEVTYPER27_EL0}}        // encoding: [0x69,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper28_el0|PMEVTYPER28_EL0}}        // encoding: [0x89,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper29_el0|PMEVTYPER29_EL0}}        // encoding: [0xa9,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{pmevtyper30_el0|PMEVTYPER30_EL0}}        // encoding: [0xc9,0xef,0x3b,0xd5]
// CHECK: mrs      x9, {{amair2_el1|AMAIR2_EL1}}        // encoding: [0x29,0xa3,0x38,0xd5]
// CHECK: mrs      x9, {{amair2_el12|AMAIR2_EL12}}        // encoding: [0x29,0xa3,0x3d,0xd5]
// CHECK: mrs      x9, {{amair2_el2|AMAIR2_EL2}}        // encoding: [0x29,0xa3,0x3c,0xd5]
// CHECK: mrs      x9, {{amair2_el3|AMAIR2_EL3}}        // encoding: [0x29,0xa3,0x3e,0xd5]
// CHECK: mrs      x9, {{mair2_el1|MAIR2_EL1}}        // encoding: [0x29,0xa2,0x38,0xd5]
// CHECK: mrs      x9, {{mair2_el12|MAIR2_EL12}}        // encoding: [0x29,0xa2,0x3d,0xd5]
// CHECK: mrs      x9, {{mair2_el2|MAIR2_EL2}}        // encoding: [0x29,0xa1,0x3c,0xd5]
// CHECK: mrs      x9, {{mair2_el3|MAIR2_EL3}}        // encoding: [0x29,0xa1,0x3e,0xd5]
// CHECK: mrs      x9, {{pire0_el1|PIRE0_EL1}}        // encoding: [0x49,0xa2,0x38,0xd5]
// CHECK: mrs      x9, {{pire0_el2|PIRE0_EL2}}        // encoding: [0x49,0xa2,0x3c,0xd5]
// CHECK: mrs      x9, {{pir_el1|PIR_EL1}}        // encoding: [0x69,0xa2,0x38,0xd5]
// CHECK: mrs      x9, {{pir_el12|PIR_EL12}}        // encoding: [0x69,0xa2,0x3d,0xd5]
// CHECK: mrs      x9, {{pir_el2|PIR_EL2}}        // encoding: [0x69,0xa2,0x3c,0xd5]
// CHECK: mrs      x9, {{pir_el3|PIR_EL3}}        // encoding: [0x69,0xa2,0x3e,0xd5]
// CHECK: mrs      x9, {{s2pir_el2|S2PIR_EL2}}        // encoding: [0xa9,0xa2,0x3c,0xd5]
// CHECK: mrs      x9, {{por_el0|POR_EL0}}        // encoding: [0x89,0xa2,0x3b,0xd5]
// CHECK: mrs      x9, {{por_el1|POR_EL1}}        // encoding: [0x89,0xa2,0x38,0xd5]
// CHECK: mrs      x9, {{por_el12|POR_EL12}}        // encoding: [0x89,0xa2,0x3d,0xd5]
// CHECK: mrs      x9, {{por_el2|POR_EL2}}        // encoding: [0x89,0xa2,0x3c,0xd5]
// CHECK: mrs      x9, {{por_el3|POR_EL3}}        // encoding: [0x89,0xa2,0x3e,0xd5]
// CHECK: mrs      x9, {{s2por_el1|S2POR_EL1}}        // encoding: [0xa9,0xa2,0x38,0xd5]
// CHECK: mrs      x9, {{sctlr2_el1|SCTLR2_EL1}}        // encoding: [0x69,0x10,0x38,0xd5]
// CHECK: mrs      x9, {{sctlr2_el12|SCTLR2_EL12}}        // encoding: [0x69,0x10,0x3d,0xd5]
// CHECK: mrs      x9, {{sctlr2_el2|SCTLR2_EL2}}        // encoding: [0x69,0x10,0x3c,0xd5]
// CHECK: mrs      x9, {{tcr2_el1|TCR2_EL1}}        // encoding: [0x69,0x20,0x38,0xd5]
// CHECK: mrs      x9, {{tcr2_el12|TCR2_EL12}}        // encoding: [0x69,0x20,0x3d,0xd5]
// CHECK: mrs      x9, {{tcr2_el2|TCR2_EL2}}        // encoding: [0x69,0x20,0x3c,0xd5]

        mrs x12, s3_7_c15_c1_5
        mrs x13, s3_2_c11_c15_7
        mrs x14, s1_3_c9_c2_1
        msr s3_0_c15_c0_0, x12
        msr s3_7_c11_c13_7, x5
        msr s1_3_c9_c2_1, x4
// CHECK: mrs     x12, {{s3_7_c15_c1_5|S3_7_C15_C1_5}}      // encoding: [0xac,0xf1,0x3f,0xd5]
// CHECK: mrs     x13, {{s3_2_c11_c15_7|S3_2_C11_C15_7}}    // encoding: [0xed,0xbf,0x3a,0xd5]
// CHECK: mrs     x14, {{s1_3_c9_c2_1|S1_3_C9_C2_1}}        // encoding: [0x2e,0x92,0x2b,0xd5]
// CHECK: msr     {{s3_0_c15_c0_0|S3_0_C15_C0_0}}, x12      // encoding: [0x0c,0xf0,0x18,0xd5]
// CHECK: msr     {{s3_7_c11_c13_7|S3_7_C11_C13_7}}, x5     // encoding: [0xe5,0xbd,0x1f,0xd5]
// CHECK: msr     {{s1_3_c9_c2_1|S1_3_C9_C2_1}}, x4         // encoding: [0x24,0x92,0x0b,0xd5]

//------------------------------------------------------------------------------
// Unconditional branch (immediate)
//------------------------------------------------------------------------------

        tbz x5, #0, somewhere
        tbz xzr, #63, elsewhere
        tbnz x5, #45, nowhere

// CHECK: tbz    w5, #0, somewhere       // encoding: [0bAAA00101,A,0b00000AAA,0x36]
// CHECK:                                //   fixup A - offset: 0, value: somewhere, kind: fixup_aarch64_pcrel_branch14
// CHECK: tbz    xzr, #63, elsewhere     // encoding: [0bAAA11111,A,0b11111AAA,0xb6]
// CHECK:                                //   fixup A - offset: 0, value: elsewhere, kind: fixup_aarch64_pcrel_branch14
// CHECK: tbnz   x5, #45, nowhere        // encoding: [0bAAA00101,A,0b01101AAA,0xb7]
// CHECK:                                //   fixup A - offset: 0, value: nowhere, kind: fixup_aarch64_pcrel_branch14


        tbnz w3, #2, there
        tbnz wzr, #31, nowhere
        tbz w5, #12, anywhere

// CHECK: tbnz    w3, #2, there           // encoding: [0bAAA00011,A,0b00010AAA,0x37]
// CHECK:                                 //   fixup A - offset: 0, value: there, kind: fixup_aarch64_pcrel_branch14
// CHECK: tbnz    wzr, #31, nowhere       // encoding: [0bAAA11111,A,0b11111AAA,0x37]
// CHECK:                                 //   fixup A - offset: 0, value: nowhere, kind: fixup_aarch64_pcrel_branch14
// CHECK: tbz     w5, #12, anywhere       // encoding: [0bAAA00101,A,0b01100AAA,0x36]
// CHECK:                                 //   fixup A - offset: 0, value: anywhere, kind: fixup_aarch64_pcrel_branch14

//------------------------------------------------------------------------------
// Unconditional branch (immediate)
//------------------------------------------------------------------------------

        b somewhere
        bl elsewhere

// CHECK: b    somewhere               // encoding: [A,A,A,0b000101AA]
// CHECK:                              //   fixup A - offset: 0, value: somewhere, kind: fixup_aarch64_pcrel_branch26
// CHECK: bl    elsewhere               // encoding: [A,A,A,0b100101AA]
// CHECK:                               //   fixup A - offset: 0, value: elsewhere, kind: fixup_aarch64_pcrel_call26

        b #4
        bl #0
        b #134217724
        bl #-134217728
// CHECK: b       #4                      // encoding: [0x01,0x00,0x00,0x14]
// CHECK: bl      #0                      // encoding: [0x00,0x00,0x00,0x94]
// CHECK: b       #134217724              // encoding: [0xff,0xff,0xff,0x15]
// CHECK: bl      #-134217728             // encoding: [0x00,0x00,0x00,0x96]

//------------------------------------------------------------------------------
// Unconditional branch (register)
//------------------------------------------------------------------------------

        br x20
        blr xzr
        ret x10
// CHECK: br       x20                        // encoding: [0x80,0x02,0x1f,0xd6]
// CHECK: blr      xzr                        // encoding: [0xe0,0x03,0x3f,0xd6]
// CHECK: ret      x10                        // encoding: [0x40,0x01,0x5f,0xd6]

        ret
        eret
        drps
// CHECK: ret                                 // encoding: [0xc0,0x03,0x5f,0xd6]
// CHECK: eret                                // encoding: [0xe0,0x03,0x9f,0xd6]
// CHECK: drps                                // encoding: [0xe0,0x03,0xbf,0xd6]

