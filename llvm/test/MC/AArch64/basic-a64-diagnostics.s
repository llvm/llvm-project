// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR --check-prefix=CHECK-ERROR-ARM64 < %t %s

//------------------------------------------------------------------------------
// Add/sub (extended register)
//------------------------------------------------------------------------------

        // Mismatched final register and extend
        add x2, x3, x5, sxtb
        add x2, x4, w2, uxtx
        add w5, w7, x9, sxtx
// CHECK-ERROR: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR:         add x2, x3, x5, sxtb
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: expected '[su]xt[bhw]' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR:         add x2, x4, w2, uxtx
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR:         add w5, w7, x9, sxtx
// CHECK-ERROR:                     ^

        // Out of range extends
        add x9, x10, w11, uxtb #-1
        add x3, x5, w7, uxtb #5
        sub x9, x15, x2, uxth #5
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR:         add x9, x10, w11, uxtb #-1
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: expected '[su]xt[bhw]' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR:         add x3, x5, w7, uxtb #5
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR:         sub x9, x15, x2, uxth #5
// CHECK-ERROR:                          ^

        // Wrong registers on normal variants
        add xzr, x3, x5, uxtx
        sub x3, xzr, w9, sxth #1
        add x1, x2, sp, uxtx
// CHECK-ERROR: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR:         add xzr, x3, x5, uxtx
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sub x3, xzr, w9, sxth #1
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR:         add x1, x2, sp, uxtx
// CHECK-ERROR:                     ^

        // Wrong registers on flag-setting variants
        adds sp, x3, w2, uxtb
        adds x3, xzr, x9, uxtx
        subs x2, x1, sp, uxtx
        adds x2, x1, sp, uxtb #2
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         adds sp, x3, w2, uxtb
// CHECK-ERROR:              ^
// CHECK-ERROR: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR:         adds x3, xzr, x9, uxtx
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR:         subs x2, x1, sp, uxtx
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR:         adds x2, x1, sp, uxtb #2
// CHECK-ERROR:                      ^

        // Amount not optional if lsl valid and used
        add sp, x5, x7, lsl
// CHECK-ERROR: error: expected #imm after shift specifier
// CHECK-ERROR:         add sp, x5, x7, lsl
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Add/sub (immediate)
//------------------------------------------------------------------------------

// Out of range immediates: more than 12 bits
        add w4, w5, #-4096
        add w5, w6, #0x1000
        add w4, w5, #-4096, lsl #12
        add w5, w6, #0x1000, lsl #12
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add w4, w5, #-4096
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-AARCH64-NEXT: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-AARCH64-NEXT:         add w5, w6, #0x1000
// CHECK-ERROR-AARCH64-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add w4, w5, #-4096, lsl #12
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add w5, w6, #0x1000, lsl #12
// CHECK-ERROR-NEXT:                     ^

// Only lsl #0 and lsl #12 are allowed
        add w2, w3, #0x1, lsl #1
        add w5, w17, #0xfff, lsl #13
        add w17, w20, #0x1000, lsl #12
        sub xsp, x34, #0x100, lsl #-1
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add w2, w3, #0x1, lsl #1
// CHECK-ERROR-NEXT:                                ^
// CHECK-ERROR-NEXT: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add w5, w17, #0xfff, lsl #13
// CHECK-ERROR-NEXT:                                   ^
// CHECK-ERROR-NEXT: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add w17, w20, #0x1000, lsl #12
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: only 'lsl #+N' valid after immediate
// CHECK-ERROR-NEXT:         sub xsp, x34, #0x100, lsl #-1
// CHECK-ERROR-NEXT:                                    ^

// Incorrect registers (w31 doesn't exist at all, and 31 decodes to sp for these).
        add w31, w20, #1234
        add wzr, w20, #0x123
        add w20, wzr, #0x321
        add wzr, wzr, #0xfff
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         add w31, w20, #1234
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         add wzr, w20, #0x123
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         add w20, wzr, #0x321
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         add wzr, wzr, #0xfff
// CHECK-ERROR-NEXT:             ^

// Mixed register classes
        add xsp, w2, #123
        sub w2, x30, #32
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         add xsp, w2, #123
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sub w2, x30, #32
// CHECK-ERROR-NEXT:                 ^

// Out of range immediate
        adds w0, w5, #0x10000
// CHECK-ERROR-AARCH64: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-AARCH64-NEXT:         adds w0, w5, #0x10000
// CHECK-ERROR-AARCH64-NEXT:                      ^

// Wn|WSP should be in second place
        adds w4, wzr, #0x123
// ...but wzr is the 31 destination
        subs wsp, w5, #123
        subs x5, xzr, #0x456, lsl #12
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adds w4, wzr, #0x123
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         subs wsp, w5, #123
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         subs x5, xzr, #0x456, lsl #12
// CHECK-ERROR-NEXT:                  ^

        // MOV alias should not accept any fiddling
        mov x2, xsp, #123
        mov wsp, w27, #0xfff, lsl #12
// CHECK-ERROR: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         mov x2, xsp, #123
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         mov wsp, w27, #0xfff, lsl #12
// CHECK-ERROR-NEXT:                       ^

        // A relocation should be provided for symbols
        add x3, x9, #variable
        add x3, x9, #variable-16
// CHECK-ERROR: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add x3, x9, #variable
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected compatible register, symbol or integer in range [0, 4095]
// CHECK-ERROR-NEXT:         add x3, x9, #variable-16
// CHECK-ERROR-NEXT:                 ^



//------------------------------------------------------------------------------
// Add-subtract (shifted register)
//------------------------------------------------------------------------------

        add wsp, w1, w2, lsr #3
        add x4, sp, x9, asr #5
        add x9, x10, x5, ror #3
// CHECK-ERROR: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add wsp, w1, w2, lsr #3
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add x4, sp, x9, asr #5
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add x9, x10, x5, ror #3
// CHECK-ERROR-NEXT:                          ^

        add w1, w2, w3, lsl #-1
        add w1, w2, w3, lsl #32
        add w1, w2, w3, lsr #-1
        add w1, w2, w3, lsr #32
        add w1, w2, w3, asr #-1
        add w1, w2, w3, asr #32
        add x1, x2, x3, lsl #-1
        add x1, x2, x3, lsl #64
        add x1, x2, x3, lsr #-1
        add x1, x2, x3, lsr #64
        add x1, x2, x3, asr #-1
        add x1, x2, x3, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         add w1, w2, w3, lsl #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add w1, w2, w3, lsl #32
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         add w1, w2, w3, lsr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add w1, w2, w3, lsr #32
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         add w1, w2, w3, asr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add w1, w2, w3, asr #32
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         add x1, x2, x3, lsl #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add x1, x2, x3, lsl #64
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         add x1, x2, x3, lsr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add x1, x2, x3, lsr #64
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         add x1, x2, x3, asr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         add x1, x2, x3, asr #64
// CHECK-ERROR-NEXT:                         ^

        adds w1, w2, w3, lsl #-1
        adds w1, w2, w3, lsl #32
        adds w1, w2, w3, lsr #-1
        adds w1, w2, w3, lsr #32
        adds w1, w2, w3, asr #-1
        adds w1, w2, w3, asr #32
        adds x1, x2, x3, lsl #-1
        adds x1, x2, x3, lsl #64
        adds x1, x2, x3, lsr #-1
        adds x1, x2, x3, lsr #64
        adds x1, x2, x3, asr #-1
        adds x1, x2, x3, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         adds w1, w2, w3, lsl #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         adds w1, w2, w3, lsl #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         adds w1, w2, w3, lsr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         adds w1, w2, w3, lsr #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         adds w1, w2, w3, asr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         adds w1, w2, w3, asr #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         adds x1, x2, x3, lsl #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         adds x1, x2, x3, lsl #64
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         adds x1, x2, x3, lsr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         adds x1, x2, x3, lsr #64
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         adds x1, x2, x3, asr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         adds x1, x2, x3, asr #64
// CHECK-ERROR-NEXT:                          ^

        sub w1, w2, w3, lsl #-1
        sub w1, w2, w3, lsl #32
        sub w1, w2, w3, lsr #-1
        sub w1, w2, w3, lsr #32
        sub w1, w2, w3, asr #-1
        sub w1, w2, w3, asr #32
        sub x1, x2, x3, lsl #-1
        sub x1, x2, x3, lsl #64
        sub x1, x2, x3, lsr #-1
        sub x1, x2, x3, lsr #64
        sub x1, x2, x3, asr #-1
        sub x1, x2, x3, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         sub w1, w2, w3, lsl #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         sub w1, w2, w3, lsl #32
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         sub w1, w2, w3, lsr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         sub w1, w2, w3, lsr #32
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         sub w1, w2, w3, asr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         sub w1, w2, w3, asr #32
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         sub x1, x2, x3, lsl #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         sub x1, x2, x3, lsl #64
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         sub x1, x2, x3, lsr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         sub x1, x2, x3, lsr #64
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         sub x1, x2, x3, asr #-1
// CHECK-ERROR-NEXT:                              ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         sub x1, x2, x3, asr #64
// CHECK-ERROR-NEXT:                         ^

        subs w1, w2, w3, lsl #-1
        subs w1, w2, w3, lsl #32
        subs w1, w2, w3, lsr #-1
        subs w1, w2, w3, lsr #32
        subs w1, w2, w3, asr #-1
        subs w1, w2, w3, asr #32
        subs x1, x2, x3, lsl #-1
        subs x1, x2, x3, lsl #64
        subs x1, x2, x3, lsr #-1
        subs x1, x2, x3, lsr #64
        subs x1, x2, x3, asr #-1
        subs x1, x2, x3, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         subs w1, w2, w3, lsl #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         subs w1, w2, w3, lsl #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         subs w1, w2, w3, lsr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         subs w1, w2, w3, lsr #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         subs w1, w2, w3, asr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         subs w1, w2, w3, asr #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         subs x1, x2, x3, lsl #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         subs x1, x2, x3, lsl #64
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         subs x1, x2, x3, lsr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         subs x1, x2, x3, lsr #64
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         subs x1, x2, x3, asr #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         subs x1, x2, x3, asr #64
// CHECK-ERROR-NEXT:                          ^

        cmn w9, w10, lsl #-1
        cmn w9, w10, lsl #32
        cmn w11, w12, lsr #-1
        cmn w11, w12, lsr #32
        cmn w19, wzr, asr #-1
        cmn wzr, wzr, asr #32
        cmn x9, x10, lsl #-1
        cmn x9, x10, lsl #64
        cmn x11, x12, lsr #-1
        cmn x11, x12, lsr #64
        cmn x19, xzr, asr #-1
        cmn xzr, xzr, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmn w9, w10, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmn w9, w10, lsl #32
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmn w11, w12, lsr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmn w11, w12, lsr #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmn w19, wzr, asr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         cmn wzr, wzr, asr #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmn x9, x10, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmn x9, x10, lsl #64
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmn x11, x12, lsr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmn x11, x12, lsr #64
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmn x19, xzr, asr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         cmn xzr, xzr, asr #64
// CHECK-ERROR-NEXT:                       ^

        cmp w9, w10, lsl #-1
        cmp w9, w10, lsl #32
        cmp w11, w12, lsr #-1
        cmp w11, w12, lsr #32
        cmp w19, wzr, asr #-1
        cmp wzr, wzr, asr #32
        cmp x9, x10, lsl #-1
        cmp x9, x10, lsl #64
        cmp x11, x12, lsr #-1
        cmp x11, x12, lsr #64
        cmp x19, xzr, asr #-1
        cmp xzr, xzr, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmp w9, w10, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmp w9, w10, lsl #32
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmp w11, w12, lsr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmp w11, w12, lsr #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmp w19, wzr, asr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         cmp wzr, wzr, asr #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmp x9, x10, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmp x9, x10, lsl #64
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmp x11, x12, lsr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]
// CHECK-ERROR-NEXT:         cmp x11, x12, lsr #64
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         cmp x19, xzr, asr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         cmp xzr, xzr, asr #64
// CHECK-ERROR-NEXT:                       ^

        neg w9, w10, lsl #-1
        neg w9, w10, lsl #32
        neg w11, w12, lsr #-1
        neg w11, w12, lsr #32
        neg w19, wzr, asr #-1
        neg wzr, wzr, asr #32
        neg x9, x10, lsl #-1
        neg x9, x10, lsl #64
        neg x11, x12, lsr #-1
        neg x11, x12, lsr #64
        neg x19, xzr, asr #-1
        neg xzr, xzr, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         neg w9, w10, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         neg w9, w10, lsl #32
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         neg w11, w12, lsr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         neg w11, w12, lsr #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         neg w19, wzr, asr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         neg wzr, wzr, asr #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         neg x9, x10, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         neg x9, x10, lsl #64
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         neg x11, x12, lsr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         neg x11, x12, lsr #64
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         neg x19, xzr, asr #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         neg xzr, xzr, asr #64
// CHECK-ERROR-NEXT:                       ^

        negs w9, w10, lsl #-1
        negs w9, w10, lsl #32
        negs w11, w12, lsr #-1
        negs w11, w12, lsr #32
        negs w19, wzr, asr #-1
        negs wzr, wzr, asr #32
        negs x9, x10, lsl #-1
        negs x9, x10, lsl #64
        negs x11, x12, lsr #-1
        negs x11, x12, lsr #64
        negs x19, xzr, asr #-1
        negs xzr, xzr, asr #64
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         negs w9, w10, lsl #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         negs w9, w10, lsl #32
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         negs w11, w12, lsr #-1
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         negs w11, w12, lsr #32
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         negs w19, wzr, asr #-1
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         negs wzr, wzr, asr #32
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         negs x9, x10, lsl #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         negs x9, x10, lsl #64
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         negs x11, x12, lsr #-1
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         negs x11, x12, lsr #64
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         negs x19, xzr, asr #-1
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         negs xzr, xzr, asr #64
// CHECK-ERROR-NEXT:                        ^

//------------------------------------------------------------------------------
// Add-subtract (shifted register)
//------------------------------------------------------------------------------

        adc wsp, w3, w5
        adc w1, wsp, w2
        adc w0, w10, wsp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        adc wsp, w3, w5
// CHECK-ERROR-NEXT:            ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adc w1, wsp, w2
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adc w0, w10, wsp
// CHECK-ERROR-NEXT:                      ^

        adc sp, x3, x5
        adc x1, sp, x2
        adc x0, x10, sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adc sp, x3, x5
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adc x1, sp, x2
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adc x0, x10, sp
// CHECK-ERROR-NEXT:                      ^

        adcs wsp, w3, w5
        adcs w1, wsp, w2
        adcs w0, w10, wsp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adcs wsp, w3, w5
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adcs w1, wsp, w2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adcs w0, w10, wsp
// CHECK-ERROR-NEXT:                       ^

        adcs sp, x3, x5
        adcs x1, sp, x2
        adcs x0, x10, sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adcs sp, x3, x5
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adcs x1, sp, x2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adcs x0, x10, sp
// CHECK-ERROR-NEXT:                       ^

        sbc wsp, w3, w5
        sbc w1, wsp, w2
        sbc w0, w10, wsp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbc wsp, w3, w5
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbc w1, wsp, w2
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbc w0, w10, wsp
// CHECK-ERROR-NEXT:                      ^

        sbc sp, x3, x5
        sbc x1, sp, x2
        sbc x0, x10, sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbc sp, x3, x5
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbc x1, sp, x2
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbc x0, x10, sp
// CHECK-ERROR-NEXT:                      ^

        sbcs wsp, w3, w5
        sbcs w1, wsp, w2
        sbcs w0, w10, wsp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbcs wsp, w3, w5
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbcs w1, wsp, w2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbcs w0, w10, wsp
// CHECK-ERROR-NEXT:                       ^

        sbcs sp, x3, x5
        sbcs x1, sp, x2
        sbcs x0, x10, sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbcs sp, x3, x5
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbcs x1, sp, x2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbcs x0, x10, sp
// CHECK-ERROR-NEXT:                       ^

        ngc wsp, w3
        ngc w9, wsp
        ngc sp, x9
        ngc x2, sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngc wsp, w3
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngc w9, wsp
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngc sp, x9
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngc x2, sp
// CHECK-ERROR-NEXT:                 ^

        ngcs wsp, w3
        ngcs w9, wsp
        ngcs sp, x9
        ngcs x2, sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngcs wsp, w3
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngcs w9, wsp
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngcs sp, x9
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ngcs x2, sp
// CHECK-ERROR-NEXT:                  ^

//------------------------------------------------------------------------------
// Logical (immediates)
//------------------------------------------------------------------------------

        and w2, w3, #4294967296
        eor w2, w3, #4294967296
        orr w2, w3, #4294967296
        ands w2, w3, #4294967296
// CHECK-ERROR: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         and w2, w3, #4294967296
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         eor w2, w3, #4294967296
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         orr w2, w3, #4294967296
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         ands w2, w3, #4294967296
// CHECK-ERROR-NEXT:                      ^

//------------------------------------------------------------------------------
// Bitfield
//------------------------------------------------------------------------------

        sbfm x3, w13, #0, #0
        sbfm w12, x9, #0, #0
        sbfm sp, x3, #3, #5
        sbfm w3, wsp, #1, #9
        sbfm x9, x5, #-1, #0
        sbfm x9, x5, #0, #-1
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfm x3, w13, #0, #0
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfm w12, x9, #0, #0
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfm sp, x3, #3, #5
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfm w3, wsp, #1, #9
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         sbfm x9, x5, #-1, #0
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         sbfm x9, x5, #0, #-1
// CHECK-ERROR-NEXT:                          ^

        sbfm w3, w5, #32, #1
        sbfm w7, w11, #19, #32
        sbfm x29, x30, #64, #0
        sbfm x10, x20, #63, #64
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         sbfm w3, w5, #32, #1
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         sbfm w7, w11, #19, #32
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         sbfm x29, x30, #64, #0
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         sbfm x10, x20, #63, #64
// CHECK-ERROR-NEXT:                             ^

        ubfm w3, w5, #32, #1
        ubfm w7, w11, #19, #32
        ubfm x29, x30, #64, #0
        ubfm x10, x20, #63, #64
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ubfm w3, w5, #32, #1
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ubfm w7, w11, #19, #32
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         ubfm x29, x30, #64, #0
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         ubfm x10, x20, #63, #64
// CHECK-ERROR-NEXT:                             ^

        bfm w3, w5, #32, #1
        bfm w7, w11, #19, #32
        bfm x29, x30, #64, #0
        bfm x10, x20, #63, #64
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         bfm w3, w5, #32, #1
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         bfm w7, w11, #19, #32
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         bfm x29, x30, #64, #0
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         bfm x10, x20, #63, #64
// CHECK-ERROR-NEXT:                             ^

        sxtb x3, x2
        sxth xzr, xzr
        sxtw x3, x5
// CHECK-ERROR-AARCH64: error: invalid operand for instruction
// CHECK-ERROR-AARCH64-NEXT:         sxtb x3, x2
// CHECK-ERROR-AARCH64-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: invalid operand for instruction
// CHECK-ERROR-AARCH64-NEXT:         sxth xzr, xzr
// CHECK-ERROR-AARCH64-NEXT:                   ^
// CHECK-ERROR-AARCH64-NEXT: error: invalid operand for instruction
// CHECK-ERROR-AARCH64-NEXT:         sxtw x3, x5
// CHECK-ERROR-AARCH64-NEXT:                  ^

        uxtb x3, x12
        uxth x5, x9
        uxtw x3, x5
        uxtb x2, sp
        uxtb sp, xzr
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         uxtb x3, x12
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         uxth x5, x9
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: invalid instruction
// CHECK-ERROR-AARCH64-NEXT:         uxtw x3, x5
// CHECK-ERROR-AARCH64-NEXT:         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         uxtb x2, sp
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         uxtb sp, xzr
// CHECK-ERROR-NEXT:              ^

        asr x3, w2, #1
        asr sp, x2, #1
        asr x25, x26, #-1
        asr x25, x26, #64
        asr w9, w8, #32
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         asr x3, w2, #1
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         asr sp, x2, #1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         asr x25, x26, #-1
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         asr x25, x26, #64
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         asr w9, w8, #32
// CHECK-ERROR-NEXT:                     ^

        sbfiz w1, w2, #0, #0
        sbfiz wsp, w9, #0, #1
        sbfiz w9, w10, #32, #1
        sbfiz w11, w12, #32, #0
        sbfiz w9, w10, #10, #23
        sbfiz x3, x5, #12, #53
        sbfiz sp, x3, #7, #6
        sbfiz w3, wsp, #10, #8
// CHECK-ERROR-AARCH64: error: expected integer in range [<lsb>, 31]
// CHECK-ERROR-ARM64: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         sbfiz w1, w2, #0, #0
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfiz wsp, w9, #0, #1
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         sbfiz w9, w10, #32, #1
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         sbfiz w11, w12, #32, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         sbfiz w9, w10, #10, #23
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         sbfiz x3, x5, #12, #53
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfiz sp, x3, #7, #6
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfiz w3, wsp, #10, #8
// CHECK-ERROR-NEXT:                   ^

        sbfx w1, w2, #0, #0
        sbfx wsp, w9, #0, #1
        sbfx w9, w10, #32, #1
        sbfx w11, w12, #32, #0
        sbfx w9, w10, #10, #23
        sbfx x3, x5, #12, #53
        sbfx sp, x3, #7, #6
        sbfx w3, wsp, #10, #8
// CHECK-ERROR-AARCH64: error: expected integer in range [<lsb>, 31]
// CHECK-ERROR-ARM64: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         sbfx w1, w2, #0, #0
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfx wsp, w9, #0, #1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         sbfx w9, w10, #32, #1
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         sbfx w11, w12, #32, #0
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: requested extract overflows register
// CHECK-ERROR-NEXT:         sbfx w9, w10, #10, #23
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: requested extract overflows register
// CHECK-ERROR-NEXT:         sbfx x3, x5, #12, #53
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfx sp, x3, #7, #6
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sbfx w3, wsp, #10, #8
// CHECK-ERROR-NEXT:                  ^

        bfi w1, w2, #0, #0
        bfi wsp, w9, #0, #1
        bfi w9, w10, #32, #1
        bfi w11, w12, #32, #0
        bfi w9, w10, #10, #23
        bfi x3, x5, #12, #53
        bfi sp, x3, #7, #6
        bfi w3, wsp, #10, #8
// CHECK-ERROR-AARCH64: error: expected integer in range [<lsb>, 31]
// CHECK-ERROR-ARM64: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         bfi w1, w2, #0, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfi wsp, w9, #0, #1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         bfi w9, w10, #32, #1
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         bfi w11, w12, #32, #0
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         bfi w9, w10, #10, #23
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         bfi x3, x5, #12, #53
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfi sp, x3, #7, #6
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfi w3, wsp, #10, #8
// CHECK-ERROR-NEXT:                 ^

        bfxil w1, w2, #0, #0
        bfxil wsp, w9, #0, #1
        bfxil w9, w10, #32, #1
        bfxil w11, w12, #32, #0
        bfxil w9, w10, #10, #23
        bfxil x3, x5, #12, #53
        bfxil sp, x3, #7, #6
        bfxil w3, wsp, #10, #8
// CHECK-ERROR-AARCH64: error: expected integer in range [<lsb>, 31]
// CHECK-ERROR-ARM64: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         bfxil w1, w2, #0, #0
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfxil wsp, w9, #0, #1
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         bfxil w9, w10, #32, #1
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         bfxil w11, w12, #32, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: requested extract overflows register
// CHECK-ERROR-NEXT:         bfxil w9, w10, #10, #23
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: requested extract overflows register
// CHECK-ERROR-NEXT:         bfxil x3, x5, #12, #53
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfxil sp, x3, #7, #6
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfxil w3, wsp, #10, #8
// CHECK-ERROR-NEXT:                   ^

        ubfiz w1, w2, #0, #0
        ubfiz wsp, w9, #0, #1
        ubfiz w9, w10, #32, #1
        ubfiz w11, w12, #32, #0
        ubfiz w9, w10, #10, #23
        ubfiz x3, x5, #12, #53
        ubfiz sp, x3, #7, #6
        ubfiz w3, wsp, #10, #8
// CHECK-ERROR-AARCH64: error: expected integer in range [<lsb>, 31]
// CHECK-ERROR-ARM64: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         ubfiz w1, w2, #0, #0
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ubfiz wsp, w9, #0, #1
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ubfiz w9, w10, #32, #1
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ubfiz w11, w12, #32, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         ubfiz w9, w10, #10, #23
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         ubfiz x3, x5, #12, #53
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ubfiz sp, x3, #7, #6
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ubfiz w3, wsp, #10, #8
// CHECK-ERROR-NEXT:                   ^

        ubfx w1, w2, #0, #0
        ubfx wsp, w9, #0, #1
        ubfx w9, w10, #32, #1
        ubfx w11, w12, #32, #0
        ubfx w9, w10, #10, #23
        ubfx x3, x5, #12, #53
        ubfx sp, x3, #7, #6
        ubfx w3, wsp, #10, #8
// CHECK-ERROR-AARCH64: error: expected integer in range [<lsb>, 31]
// CHECK-ERROR-ARM64: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         ubfx w1, w2, #0, #0
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ubfx wsp, w9, #0, #1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ubfx w9, w10, #32, #1
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ubfx w11, w12, #32, #0
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: requested extract overflows register
// CHECK-ERROR-NEXT:         ubfx w9, w10, #10, #23
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: requested extract overflows register
// CHECK-ERROR-NEXT:         ubfx x3, x5, #12, #53
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ubfx sp, x3, #7, #6
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ubfx w3, wsp, #10, #8
// CHECK-ERROR-NEXT:                  ^

        bfc wsp, #3, #6
        bfc w4, #2, #31
        bfc sp, #0, #1
        bfc x6, #0, #0
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        bfc wsp, #3, #6
// CHECK-ERROR-NEXT:            ^
// CHECK-ERROR-NEXT: error: requested insert overflows register
// CHECK-ERROR-NEXT:         bfc w4, #2, #31
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bfc sp, #0, #1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected integer in range [1, 32]
// CHECK-ERROR-NEXT:         bfc x6, #0, #0
// CHECK-ERROR-NEXT:                     ^

//------------------------------------------------------------------------------
// Compare & branch (immediate)
//------------------------------------------------------------------------------

        cbnz wsp, lbl
        cbz  sp, lbl
        cbz  x3, x5
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:           cbnz wsp, lbl
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:           cbz sp, lbl
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           cbz x3, x5
// CHECK-ERROR-NEXT:                   ^

        cbz w20, #1048576
        cbnz xzr, #-1048580
        cbz x29, #1
// CHECK-ERROR: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           cbz w20, #1048576
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           cbnz xzr, #-1048580
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           cbz x29, #1
// CHECK-ERROR-NEXT:                    ^

//------------------------------------------------------------------------------
// Conditional branch (immediate)
//------------------------------------------------------------------------------

        b.zf lbl
// CHECK-ERROR: error: invalid condition code
// CHECK-ERROR-NEXT:           b.zf lbl
// CHECK-ERROR-NEXT:             ^

        b.eq #1048576
        b.ge #-1048580
        b.cc #1
// CHECK-ERROR: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           b.eq #1048576
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           b.ge #-1048580
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:           b.cc #1
// CHECK-ERROR-NEXT:                ^

//------------------------------------------------------------------------------
// Conditional compare (immediate)
//------------------------------------------------------------------------------

        ccmp wsp, #4, #2, ne
        ccmp w25, #-1, #15, hs
        ccmp w3, #32, #0, ge
        ccmp w19, #5, #-1, lt
        ccmp w20, #7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmp wsp, #4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmp w25, #-1, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmp w3, #32, #0, ge
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp w19, #5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp w20, #7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        ccmp sp, #4, #2, ne
        ccmp x25, #-1, #15, hs
        ccmp x3, #32, #0, ge
        ccmp x19, #5, #-1, lt
        ccmp x20, #7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmp sp, #4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmp x25, #-1, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmp x3, #32, #0, ge
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp x19, #5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp x20, #7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        ccmn wsp, #4, #2, ne
        ccmn w25, #-1, #15, hs
        ccmn w3, #32, #0, ge
        ccmn w19, #5, #-1, lt
        ccmn w20, #7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmn wsp, #4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmn w25, #-1, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmn w3, #32, #0, ge
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn w19, #5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn w20, #7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        ccmn sp, #4, #2, ne
        ccmn x25, #-1, #15, hs
        ccmn x3, #32, #0, ge
        ccmn x19, #5, #-1, lt
        ccmn x20, #7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmn sp, #4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmn x25, #-1, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmn x3, #32, #0, ge
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn x19, #5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn x20, #7, #16, hs
// CHECK-ERROR-NEXT:                      ^

//------------------------------------------------------------------------------
// Conditional compare (register)
//------------------------------------------------------------------------------

        ccmp wsp, w4, #2, ne
        ccmp w3, wsp, #0, ge
        ccmp w19, w5, #-1, lt
        ccmp w20, w7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmp wsp, w4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmp w3, wsp, #0, ge
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp w19, w5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp w20, w7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        ccmp sp, x4, #2, ne
        ccmp x25, sp, #15, hs
        ccmp x19, x5, #-1, lt
        ccmp x20, x7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmp sp, x4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmp x25, sp, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp x19, x5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmp x20, x7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        ccmn wsp, w4, #2, ne
        ccmn w25, wsp, #15, hs
        ccmn w19, w5, #-1, lt
        ccmn w20, w7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmn wsp, w4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmn w25, wsp, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn w19, w5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn w20, w7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        ccmn sp, x4, #2, ne
        ccmn x25, sp, #15, hs
        ccmn x19, x5, #-1, lt
        ccmn x20, x7, #16, hs
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ccmn sp, x4, #2, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        ccmn x25, sp, #15, hs
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn x19, x5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        ccmn x20, x7, #16, hs
// CHECK-ERROR-NEXT:                      ^

//------------------------------------------------------------------------------
// Conditional select
//------------------------------------------------------------------------------

        csel w4, wsp, w9, eq
        csel wsp, w2, w3, ne
        csel w10, w11, wsp, ge
        csel w1, w2, w3, #3
        csel x4, sp, x9, eq
        csel sp, x2, x3, ne
        csel x10, x11, sp, ge
        csel x1, x2, x3, #3
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csel w4, wsp, w9, eq
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csel wsp, w2, w3, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csel w10, w11, wsp, ge
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected AArch64 condition code
// CHECK-ERROR-NEXT:        csel w1, w2, w3, #3
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csel x4, sp, x9, eq
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csel sp, x2, x3, ne
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csel x10, x11, sp, ge
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: expected AArch64 condition code
// CHECK-ERROR-NEXT:        csel x1, x2, x3, #3
// CHECK-ERROR-NEXT:                         ^

        csinc w20, w21, wsp, mi
        csinc sp, x30, x29, eq
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csinc w20, w21, wsp, mi
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csinc sp, x30, x29, eq
// CHECK-ERROR-NEXT:              ^

        csinv w20, wsp, wsp, mi
        csinv sp, x30, x29, le
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csinv w20, wsp, wsp, mi
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csinv sp, x30, x29, le
// CHECK-ERROR-NEXT:              ^

        csneg w20, w21, wsp, mi
        csneg x0, sp, x29, le
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csneg w20, w21, wsp, mi
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csneg x0, sp, x29, le
// CHECK-ERROR-NEXT:                  ^

        cset wsp, lt
        csetm sp, ge
        cset w1, al
        csetm x6, nv
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cset wsp, lt
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        csetm sp, ge
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: condition codes AL and NV are invalid for this instruction
// CHECK-ERROR-NEXT:        cset w1, al
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: condition codes AL and NV are invalid for this instruction
// CHECK-ERROR-NEXT:        csetm x6, nv
// CHECK-ERROR-NEXT:                    ^

        cinc w3, wsp, ne
        cinc sp, x9, eq
        cinc x2, x0, nv
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cinc w3, wsp, ne
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cinc sp, x9, eq
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: condition codes AL and NV are invalid for this instruction
// CHECK-ERROR-NEXT:        cinc x2, x0, nv
// CHECK-ERROR-NEXT:                       ^

        cinv w3, wsp, ne
        cinv sp, x9, eq
        cinv w8, x7, nv
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cinv w3, wsp, ne
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cinv sp, x9, eq
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: condition codes AL and NV are invalid for this instruction
// CHECK-ERROR-NEXT:        cinv w8, x7, nv
// CHECK-ERROR-NEXT:                       ^

        cneg w3, wsp, ne
        cneg sp, x9, eq
        cneg x4, x5, al
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cneg w3, wsp, ne
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        cneg sp, x9, eq
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: condition codes AL and NV are invalid for this instruction
// CHECK-ERROR-NEXT:        cneg x4, x5, al
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Data Processing (1 source)
//------------------------------------------------------------------------------
        rbit x23, w2
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     rbit x23, w2

        cls sp, x2
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     cls sp, x2

        clz wsp, w3
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     clz wsp, w3

//------------------------------------------------------------------------------
// Data Processing (2 sources)
//------------------------------------------------------------------------------
        udiv x23, w2, x18
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     udiv x23, w2, x18

        lsl sp, x2, x4
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     lsl sp, x2, x4

        asr wsp, w3, w9
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     asr wsp, w3, w9

//------------------------------------------------------------------------------
// Data Processing (3 sources)
//------------------------------------------------------------------------------

        madd sp, x3, x9, x10
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR-NEXT:     madd sp, x3, x9, x10

//------------------------------------------------------------------------------
// Exception generation
//------------------------------------------------------------------------------
        svc #-1
        hlt #65536
        dcps4 #43
        dcps4
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         svc #-1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         hlt #65536
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{invalid instruction|unrecognized instruction mnemonic}}
// CHECK-ERROR-NEXT:         dcps4 #43
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR-NEXT: error: {{invalid instruction|unrecognized instruction mnemonic}}
// CHECK-ERROR-NEXT:         dcps4
// CHECK-ERROR-NEXT:         ^

//------------------------------------------------------------------------------
// Extract (immediate)
//------------------------------------------------------------------------------

        extr w2, w20, w30, #-1
        extr w9, w19, w20, #32
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         extr w2, w20, w30, #-1
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         extr w9, w19, w20, #32
// CHECK-ERROR-NEXT:                            ^

        extr x10, x15, x20, #-1
        extr x20, x25, x30, #64
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         extr x10, x15, x20, #-1
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         extr x20, x25, x30, #64
// CHECK-ERROR-NEXT:                             ^

        ror w9, w10, #32
        ror x10, x11, #64
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:         ror w9, w10, #32
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:         ror x10, x11, #64
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Floating-point compare
//------------------------------------------------------------------------------

        fcmp s3, d2
// CHECK-ERROR-AARCH64: error: expected floating-point constant #0.0
// CHECK-ERROR-ARM64: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fcmp s3, d2
// CHECK-ERROR-NEXT:                  ^

        fcmp s9, #-0.0
        fcmp d3, #-0.0
        fcmp s1, #1.0
        fcmpe s30, #-0.0
// CHECK-ERROR: error: expected floating-point constant #0.0
// CHECK-ERROR-NEXT:         fcmp s9, #-0.0
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected floating-point constant #0.0
// CHECK-ERROR-NEXT:         fcmp d3, #-0.0
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected floating-point constant #0.0
// CHECK-ERROR-NEXT:         fcmp s1, #1.0
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected floating-point constant #0.0
// CHECK-ERROR-NEXT:         fcmpe s30, #-0.0
// CHECK-ERROR-NEXT:                    ^

//------------------------------------------------------------------------------
// Floating-point conditional compare
//------------------------------------------------------------------------------

        fccmp s19, s5, #-1, lt
        fccmp s20, s7, #16, hs
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmp s19, s5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmp s20, s7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        fccmp d19, d5, #-1, lt
        fccmp d20, d7, #16, hs
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmp d19, d5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmp d20, d7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        fccmpe s19, s5, #-1, lt
        fccmpe s20, s7, #16, hs
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmpe s19, s5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmpe s20, s7, #16, hs
// CHECK-ERROR-NEXT:                      ^

        fccmpe d19, d5, #-1, lt
        fccmpe d20, d7, #16, hs
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmpe d19, d5, #-1, lt
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:        fccmpe d20, d7, #16, hs
// CHECK-ERROR-NEXT:                      ^

//------------------------------------------------------------------------------
// Floating-point conditional compare
//------------------------------------------------------------------------------

        fcsel q3, q20, q9, pl
        fcsel h9, h10, h11, mi
        fcsel b9, b10, b11, mi
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fcsel q3, q20, q9, pl
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: instruction requires: fullfp16
// CHECK-ERROR-NEXT:         fcsel h9, h10, h11, mi
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fcsel b9, b10, b11, mi
// CHECK-ERROR-NEXT:               ^

//------------------------------------------------------------------------------
// Floating-point data-processing (1 source)
//------------------------------------------------------------------------------

        fmov d0, s3
        fcvt d0, d1
// CHECK-ERROR: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov d0, s3
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:           fcvt d0, d1
// CHECK-ERROR-NEXT:                    ^


//------------------------------------------------------------------------------
// Floating-point data-processing (2 sources)
//------------------------------------------------------------------------------

        fadd s0, d3, d7
        fmaxnm d3, s19, d12
        fnmul d1, d9, s18
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:           fadd s0, d3, d7
// CHECK-ERROR-NEXT: ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:           fmaxnm d3, s19, d12
// CHECK-ERROR-NEXT: ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:           fnmul d1, d9, s18
// CHECK-ERROR-NEXT: ^

//------------------------------------------------------------------------------
// Floating-point data-processing (3 sources)
//------------------------------------------------------------------------------

        fmadd b3, b4, b5, b6
        fmsub h1, h2, h3, h4
        fnmadd q3, q5, q6, q7
        fnmsub s2, s4, d5, h9
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fmadd b3, b4, b5, b6
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: instruction requires: fullfp16
// CHECK-ERROR-NEXT:         fmsub h1, h2, h3, h4
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fnmadd q3, q5, q6, q7
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fnmsub s2, s4, d5, h9
// CHECK-ERROR-NEXT:                ^

//------------------------------------------------------------------------------
// Floating-point conditional compare
//------------------------------------------------------------------------------

        fcvtzs w13, s31, #0
        fcvtzs w19, s20, #33
        fcvtzs wsp, s19, #14
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR-NEXT:        fcvtzs w13, s31, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR-NEXT:        fcvtzs w19, s20, #33
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        fcvtzs wsp, s19, #14
// CHECK-ERROR-NEXT:               ^

        fcvtzs x13, s31, #0
        fcvtzs x19, s20, #65
        fcvtzs sp, s19, #14
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR-NEXT:        fcvtzs x13, s31, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR-NEXT:        fcvtzs x19, s20, #65
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        fcvtzs sp, s19, #14
// CHECK-ERROR-NEXT:               ^

        fcvtzu w13, s31, #0
        fcvtzu w19, s20, #33
        fcvtzu wsp, s19, #14
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR-NEXT:        fcvtzu w13, s31, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR-NEXT:        fcvtzu w19, s20, #33
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        fcvtzu wsp, s19, #14
// CHECK-ERROR-NEXT:               ^

        fcvtzu x13, s31, #0
        fcvtzu x19, s20, #65
        fcvtzu sp, s19, #14
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR-NEXT:        fcvtzu x13, s31, #0
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR-NEXT:        fcvtzu x19, s20, #65
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        fcvtzu sp, s19, #14
// CHECK-ERROR-NEXT:               ^

        scvtf w13, s31, #0
        scvtf w19, s20, #33
        scvtf wsp, s19, #14
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        scvtf w13, s31, #0
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        scvtf w19, s20, #33
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        scvtf wsp, s19, #14
// CHECK-ERROR-NEXT:              ^

        scvtf x13, s31, #0
        scvtf x19, s20, #65
        scvtf sp, s19, #14
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        scvtf x13, s31, #0
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        scvtf x19, s20, #65
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        scvtf sp, s19, #14
// CHECK-ERROR-NEXT:              ^

        ucvtf w13, s31, #0
        ucvtf w19, s20, #33
        ucvtf wsp, s19, #14
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ucvtf w13, s31, #0
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ucvtf w19, s20, #33
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ucvtf wsp, s19, #14
// CHECK-ERROR-NEXT:              ^

        ucvtf x13, s31, #0
        ucvtf x19, s20, #65
        ucvtf sp, s19, #14
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ucvtf x13, s31, #0
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ucvtf x19, s20, #65
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ucvtf sp, s19, #14
// CHECK-ERROR-NEXT:              ^

//------------------------------------------------------------------------------
// Floating-point immediate
//------------------------------------------------------------------------------
        ;; Exponent too large
        fmov d3, #0.0625
        fmov s2, #32.0
        fmov s2, #32
        fmov v0.4s, #-32
// CHECK-ERROR: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov d3, #0.0625
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov s2, #32.0
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov s2, #32
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov v0.4s, #-32
// CHECK-ERROR-NEXT:                       ^

        ;; Fraction too precise
        fmov s9, #1.03125
        fmov s28, #1.96875
// CHECK-ERROR: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov s9, #1.03125
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov s28, #1.96875
// CHECK-ERROR-NEXT:                     ^

        ;; Explicitly encoded value too large
        fmov s15, #0x100
// CHECK-ERROR: error: encoded floating point value out of range
// CHECK-ERROR-NEXT:           fmov s15, #0x100
// CHECK-ERROR-NEXT:                     ^

        ;; Not possible to fmov ZR to a whole vector
        fmov v0.4s, #0.0
// CHECK-ERROR: error: expected compatible register or floating-point constant
// CHECK-ERROR-NEXT:           fmov v0.4s, #0.0
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Floating-point <-> integer conversion
//------------------------------------------------------------------------------

        fmov x3, v0.d[0]
        fmov v29.1d[1], x2
        fmov x7, v0.d[2]
        fcvtns sp, s5
        scvtf s6, wsp
// CHECK-ERROR: error: expected lane specifier '[1]'
// CHECK-ERROR-NEXT:         fmov x3, v0.d[0]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-AARCH64-NEXT: error: lane number incompatible with layout
// CHECK-ERROR-ARM64-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT: fmov v29.1d[1], x2
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-AARCH64-NEXT: error: lane number incompatible with layout
// CHECK-ERROR-ARM64-NEXT: error: expected lane specifier '[1]'
// CHECK-ERROR-NEXT: fmov x7, v0.d[2]
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         fcvtns sp, s5
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         scvtf s6, wsp
// CHECK-ERROR-NEXT:                   ^

//------------------------------------------------------------------------------
// Load-register (literal)
//------------------------------------------------------------------------------

        ldr sp, some_label
        ldrsw w3, somewhere
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldr sp, some_label
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrsw w3, somewhere
// CHECK-ERROR-NEXT:               ^

        ldrsw x2, #1048576
        ldr q0, #-1048580
        ldr x0, #2
// CHECK-ERROR: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldrsw x2, #1048576
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr q0, #-1048580
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr x0, #2
// CHECK-ERROR-NEXT:                 ^

//------------------------------------------------------------------------------
// Load/store exclusive
//------------------------------------------------------------------------------

       stxrb w2, w3, [x4, #20]
       stlxrh w10, w11, [w2]
// CHECK-ERROR-AARCH64: error: expected '#0'
// CHECK-ERROR-ARM64: error: index must be absent or #0
// CHECK-ERROR-NEXT:         stxrb w2, w3, [x4, #20]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stlxrh w10, w11, [w2]
// CHECK-ERROR-NEXT:                           ^

       stlxr  x20, w21, [sp]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stlxr  x20, w21, [sp]
// CHECK-ERROR-NEXT:                ^

       ldxr   sp, [sp]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldxr   sp, [sp]
// CHECK-ERROR-NEXT:                ^

       stxp x1, x2, x3, [x4]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stxp x1, x2,  x3, [x4]
// CHECK-ERROR-NEXT:              ^

       stlxp w5, x1, w4, [x5]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stlxp w5, x1, w4, [x5]
// CHECK-ERROR-NEXT:                       ^

       stlxp w17, w6, x7, [x22]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stlxp w17, w6, x7, [x22]
// CHECK-ERROR-NEXT:                        ^

//------------------------------------------------------------------------------
// Load/store (unscaled immediate)
//------------------------------------------------------------------------------

        ldurb w2, [sp, #256]
        sturh w17, [x1, #256]
        ldursw x20, [x1, #256]
        ldur x12, [sp, #256]
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:        ldurb w2, [sp, #256]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         sturh w17, [x1, #256]
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldursw x20, [x1, #256]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldur x12, [sp, #256]
// CHECK-ERROR-NEXT:                   ^

        stur h2, [x2, #-257]
        stur b2, [x2, #-257]
        ldursb x9, [sp, #-257]
        ldur w2, [x30, #-257]
        stur q9, [x20, #-257]
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         stur h2, [x2, #-257]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         stur b2, [x2, #-257]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldursb x9, [sp, #-257]
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldur w2, [x30, #-257]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         stur q9, [x20, #-257]
// CHECK-ERROR-NEXT:                  ^

        prfum pstl3strm, [xzr]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         prfum pstl3strm, [xzr]
// CHECK-ERROR-NEXT:                           ^

//------------------------------------------------------------------------------
// Load-store register (immediate post-indexed)
//------------------------------------------------------------------------------
        ldr x3, [x4, #25], #0
        ldr x4, [x9, #0], #4
// CHECK-ERROR-AARCH64: error: {{expected symbolic reference or integer|index must be a multiple of 8}} in range [0, 32760]
// CHECK-ERROR-ARM64: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr x3, [x4, #25], #0
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-AARCH64-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-AARCH64-NEXT:         ldr x4, [x9, #0], #4
// CHECK-ERROR-AARCH64-NEXT:                           ^

        strb w1, [x19], #256
        strb w9, [sp], #-257
        strh w1, [x19], #256
        strh w9, [sp], #-257
        str w1, [x19], #256
        str w9, [sp], #-257
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         strb w1, [x19], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         strb w9, [sp], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         strh w1, [x19], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         strh w9, [sp], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str w1, [x19], #256
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str w9, [sp], #-257
// CHECK-ERROR-NEXT:                       ^

        ldrb w1, [x19], #256
        ldrb w9, [sp], #-257
        ldrh w1, [x19], #256
        ldrh w9, [sp], #-257
        ldr w1, [x19], #256
        ldr w9, [sp], #-257
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrb w1, [x19], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrb w9, [sp], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrh w1, [x19], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrh w9, [sp], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr w1, [x19], #256
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr w9, [sp], #-257
// CHECK-ERROR-NEXT:                       ^

        ldrsb x2, [x3], #256
        ldrsb x22, [x13], #-257
        ldrsh x2, [x3], #256
        ldrsh x22, [x13], #-257
        ldrsw x2, [x3], #256
        ldrsw x22, [x13], #-257
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsb x2, [x3], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsb x22, [x13], #-257
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsh x2, [x3], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsh x22, [x13], #-257
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsw x2, [x3], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsw x22, [x13], #-257
// CHECK-ERROR-NEXT:                           ^

        ldrsb w2, [x3], #256
        ldrsb w22, [x13], #-257
        ldrsh w2, [x3], #256
        ldrsh w22, [x13], #-257
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsb w2, [x3], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsb w22, [x13], #-257
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsh w2, [x3], #256
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsh w22, [x13], #-257
// CHECK-ERROR-NEXT:                           ^

        str b3, [x3], #256
        str b3, [x13], #-257
        str h3, [x3], #256
        str h3, [x13], #-257
        str s3, [x3], #256
        str s3, [x13], #-257
        str d3, [x3], #256
        str d3, [x13], #-257
        str q3, [x3], #256
        str q3, [x13], #-257
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str b3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str b3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str h3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str h3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str s3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str s3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str d3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str d3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str q3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str q3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^

        ldr b3, [x3], #256
        ldr b3, [x13], #-257
        ldr h3, [x3], #256
        ldr h3, [x13], #-257
        ldr s3, [x3], #256
        ldr s3, [x13], #-257
        ldr d3, [x3], #256
        ldr d3, [x13], #-257
        ldr q3, [x3], #256
        ldr q3, [x13], #-257
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr b3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr b3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr h3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr h3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr s3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr s3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr d3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr d3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr q3, [x3], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr q3, [x13], #-257
// CHECK-ERROR-NEXT:                        ^

//------------------------------------------------------------------------------
// Load-store register (immediate pre-indexed)
//------------------------------------------------------------------------------

        ldr x3, [x4]!
// CHECK-ERROR: error:
// CHECK-ERROR-NEXT:         ldr x3, [x4]!
// CHECK-ERROR-NEXT:                     ^

        strb w1, [x19, #256]!
        strb w9, [sp, #-257]!
        strh w1, [x19, #256]!
        strh w9, [sp, #-257]!
        str w1, [x19, #256]!
        str w9, [sp, #-257]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         strb w1, [x19, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         strb w9, [sp, #-257]!
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         strh w1, [x19, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         strh w9, [sp, #-257]!
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         str w1, [x19, #256]!
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str w9, [sp, #-257]!
// CHECK-ERROR-NEXT:                 ^

        ldrb w1, [x19, #256]!
        ldrb w9, [sp, #-257]!
        ldrh w1, [x19, #256]!
        ldrh w9, [sp, #-257]!
        ldr w1, [x19, #256]!
        ldr w9, [sp, #-257]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrb w1, [x19, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrb w9, [sp, #-257]!
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrh w1, [x19, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrh w9, [sp, #-257]!
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr w1, [x19, #256]!
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr w9, [sp, #-257]!
// CHECK-ERROR-NEXT:                 ^

        ldrsb x2, [x3, #256]!
        ldrsb x22, [x13, #-257]!
        ldrsh x2, [x3, #256]!
        ldrsh x22, [x13, #-257]!
        ldrsw x2, [x3, #256]!
        ldrsw x22, [x13, #-257]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrsb x2, [x3, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsb x22, [x13, #-257]!
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrsh x2, [x3, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsh x22, [x13, #-257]!
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldrsw x2, [x3, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsw x22, [x13, #-257]!
// CHECK-ERROR-NEXT:                    ^

        ldrsb w2, [x3, #256]!
        ldrsb w22, [x13, #-257]!
        ldrsh w2, [x3, #256]!
        ldrsh w22, [x13, #-257]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrsb w2, [x3, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsb w22, [x13, #-257]!
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrsh w2, [x3, #256]!
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrsh w22, [x13, #-257]!
// CHECK-ERROR-NEXT:                    ^

        str b3, [x3, #256]!
        str b3, [x13, #-257]!
        str h3, [x3, #256]!
        str h3, [x13, #-257]!
        str s3, [x3, #256]!
        str s3, [x13, #-257]!
        str d3, [x3, #256]!
        str d3, [x13, #-257]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         str b3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str b3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         str h3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str h3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         str s3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str s3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         str d3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str d3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^

        ldr b3, [x3, #256]!
        ldr b3, [x13, #-257]!
        ldr h3, [x3, #256]!
        ldr h3, [x13, #-257]!
        ldr s3, [x3, #256]!
        ldr s3, [x13, #-257]!
        ldr d3, [x3, #256]!
        ldr d3, [x13, #-257]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldr b3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr b3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldr h3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr h3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr s3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr s3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr d3, [x3, #256]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr d3, [x13, #-257]!
// CHECK-ERROR-NEXT:                 ^

//------------------------------------------------------------------------------
// Load/store (unprivileged)
//------------------------------------------------------------------------------

        ldtrb w2, [sp, #256]
        sttrh w17, [x1, #256]
        ldtrsw x20, [x1, #256]
        ldtr x12, [sp, #256]
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:        ldtrb w2, [sp, #256]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         sttrh w17, [x1, #256]
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldtrsw x20, [x1, #256]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldtr x12, [sp, #256]
// CHECK-ERROR-NEXT:                   ^

        sttr h2, [x2, #-257]
        sttr b2, [x2, #-257]
        ldtrsb x9, [sp, #-257]
        ldtr w2, [x30, #-257]
        sttr q9, [x20, #-257]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sttr h2, [x2, #-257]
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sttr b2, [x2, #-257]
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldtrsb x9, [sp, #-257]
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldtr w2, [x30, #-257]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         sttr q9, [x20, #-257]
// CHECK-ERROR-NEXT:                  ^


//------------------------------------------------------------------------------
// Load/store (unsigned immediate)
//------------------------------------------------------------------------------

//// Out of range immediates
        ldr q0, [x11, #65536]
        ldr x0, [sp, #32768]
        ldr w0, [x4, #16384]
        ldrh w2, [x21, #8192]
        ldrb w3, [x12, #4096]
// CHECK-ERROR: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr q0, [x11, #65536]
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr x0, [sp, #32768]
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldr w0, [x4, #16384]
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrh w2, [x21, #8192]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         ldrb w3, [x12, #4096]
// CHECK-ERROR-NEXT:                  ^

//// Misaligned addresses
        ldr w0, [x0, #2]
        ldrsh w2, [x0, #123]
        str q0, [x0, #8]
// CHECK-ERROR-AARCH64: error: too few operands for instruction
// CHECK-ERROR-AARCH64-NEXT:         ldr w0, [x0, #2]
// CHECK-ERROR-AARCH64-NEXT:                 ^
// CHECK-ERROR-AARCH64-NEXT: error: too few operands for instruction
// CHECK-ERROR-AARCH64-NEXT:         ldrsh w2, [x0, #123]
// CHECK-ERROR-AARCH64-NEXT:                   ^
// CHECK-ERROR-AARCH64-NEXT: error: too few operands for instruction
// CHECK-ERROR-AARCH64-NEXT:         str q0, [x0, #8]
// CHECK-ERROR-AARCH64-NEXT:                 ^

//// 32-bit addresses
        ldr w0, [w20]
        ldrsh x3, [wsp]
// CHECK-ERROR: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         ldr w0, [w20]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldrsh x3, [wsp]
// CHECK-ERROR-NEXT:                    ^

//// Store things
        strb w0, [wsp]
        strh w31, [x23, #1]
        str x5, [x22, #12]
        str w7, [x12, #16384]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT: strb w0, [wsp]
// CHECK-ERROR-NEXT:           ^
// CHECK-ERROR-AARCH64: error: invalid operand for instruction
// CHECK-ERROR-AARCH64-NEXT:         strh w31, [x23, #1]
// CHECK-ERROR-AARCH64-NEXT:              ^
// CHECK-ERROR-AARCH64-NEXT: error: too few operands for instruction
// CHECK-ERROR-AARCH64-NEXT:         str x5, [x22, #12]
// CHECK-ERROR-AARCH64-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|index must be an}} integer in range [-256, 255]
// CHECK-ERROR-NEXT:         str w7, [x12, #16384]
// CHECK-ERROR-NEXT:                 ^

//// Bad PRFMs
        prfm #-1, [sp]
        prfm #32, [sp, #8]
        prfm pldl1strm, [w3, #8]
        prfm wibble, [sp]
// CHECK-ERROR-AARCH64: error: Invalid immediate for instruction
// CHECK-ERROR-ARM64: error: prefetch operand out of range, [0,31] expected
// CHECK-ERROR-NEXT:        prfm #-1, [sp]
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-AARCH64-NEXT: error: Invalid immediate for instruction
// CHECK-ERROR-ARM64-NEXT: error: prefetch operand out of range, [0,31] expected
// CHECK-ERROR-NEXT:        prfm #32, [sp, #8]
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:        prfm pldl1strm, [w3, #8]
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-AARCH64-NEXT: error: operand specifier not recognised
// CHECK-ERROR-ARM64-NEXT: error: pre-fetch hint expected
// CHECK-ERROR-NEXT:        prfm wibble, [sp]
// CHECK-ERROR-NEXT:             ^

//------------------------------------------------------------------------------
// Load/store register (register offset)
//------------------------------------------------------------------------------

        ldr w3, [xzr, x3]
        ldr w4, [x0, x4, lsl]
        ldr w9, [x5, x5, uxtw]
        ldr w10, [x6, x9, sxtw #2]
        ldr w11, [x7, w2, lsl #2]
        ldr w12, [x8, w1, sxtx]
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:        ldr w3, [xzr, x3]
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected #imm after shift specifier
// CHECK-ERROR-NEXT:         ldr w4, [x0, x4, lsl]
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: expected 'lsl' or 'sxtx' with optional shift of #0 or #2
// CHECK-ERROR-NEXT:         ldr w9, [x5, x5, uxtw]
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected 'lsl' or 'sxtx' with optional shift of #0 or #2
// CHECK-ERROR-NEXT:         ldr w10, [x6, x9, sxtw #2]
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #2
// CHECK-ERROR-NEXT:         ldr w11, [x7, w2, lsl #2]
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #2
// CHECK-ERROR-NEXT:         ldr w12, [x8, w1, sxtx]
// CHECK-ERROR-NEXT:                           ^

        ldrsb w9, [x4, x2, lsl #-1]
        strb w9, [x4, x2, lsl #1]
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         ldrsb w9, [x4, x2, lsl #-1]
// CHECK-ERROR-NEXT:                                 ^
// CHECK-ERROR-NEXT: error: expected 'lsl' or 'sxtx' with optional shift of #0
// CHECK-ERROR-NEXT:         strb w9, [x4, x2, lsl #1]
// CHECK-ERROR-NEXT:                  ^

        ldrsh w9, [x4, x2, lsl #-1]
        ldr h13, [x4, w2, uxtw #2]
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         ldrsh w9, [x4, x2, lsl #-1]
// CHECK-ERROR-NEXT:                                 ^
// CHECK-ERROR-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #1
// CHECK-ERROR-NEXT:         ldr h13, [x4, w2, uxtw #2]
// CHECK-ERROR-NEXT:                           ^

        str w9, [x5, w9, sxtw #-1]
        str s3, [sp, w9, uxtw #1]
        ldrsw x9, [x15, x4, sxtx #3]
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         str w9, [x5, w9, sxtw #-1]
// CHECK-ERROR-NEXT:                                ^
// CHECK-ERROR-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #2
// CHECK-ERROR-NEXT:         str s3, [sp, w9, uxtw #1]
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected 'lsl' or 'sxtx' with optional shift of #0 or #2
// CHECK-ERROR-NEXT:         ldrsw x9, [x15, x4, sxtx #3]
// CHECK-ERROR-NEXT:                             ^

        str xzr, [x5, x9, sxtx #-1]
        prfm pldl3keep, [sp, x20, lsl #2]
        ldr d3, [x20, wzr, uxtw #4]
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         str xzr, [x5, x9, sxtx #-1]
// CHECK-ERROR-NEXT:                                 ^
// CHECK-ERROR-NEXT: error: expected 'lsl' or 'sxtx' with optional shift of #0 or #3
// CHECK-ERROR-NEXT:         prfm pldl3keep, [sp, x20, lsl #2]
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #3
// CHECK-ERROR-NEXT:         ldr d3, [x20, wzr, uxtw #4]
// CHECK-ERROR-NEXT:                 ^

        ldr q5, [sp, x2, lsl #-1]
        ldr q10, [x20, w4, uxtw #2]
        str q21, [x20, w4, uxtw #5]
// CHECK-ERROR-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         ldr q5, [sp, x2, lsl #-1]
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-AARCH64-NEXT: error: expected 'lsl' or 'sxtw' with optional shift of #0 or #4
// CHECK-ERROR-ARM64-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #4
// CHECK-ERROR-NEXT:         ldr q10, [x20, w4, uxtw #2]
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: expected 'lsl' or 'sxtw' with optional shift of #0 or #4
// CHECK-ERROR-ARM64-NEXT: error: expected 'uxtw' or 'sxtw' with optional shift of #0 or #4
// CHECK-ERROR-NEXT:         str q21, [x20, w4, uxtw #5]
// CHECK-ERROR-NEXT:                  ^

//------------------------------------------------------------------------------
// Load/store register pair (offset)
//------------------------------------------------------------------------------
        ldp w3, w2, [x4, #1]
        stp w1, w2, [x3, #253]
        stp w9, w10, [x5, #256]
        ldp w11, w12, [x9, #-260]
        stp wsp, w9, [sp]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp w3, w2, [x4, #1]
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp w1, w2, [x3, #253]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp w9, w10, [x5, #256]
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp w11, w12, [x9, #-260]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stp wsp, w9, [sp]
// CHECK-ERROR-NEXT:             ^

        ldpsw x9, x2, [sp, #2]
        ldpsw x1, x2, [x10, #256]
        ldpsw x3, x4, [x11, #-260]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x9, x2, [sp, #2]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x1, x2, [x10, #256]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x3, x4, [x11, #-260]
// CHECK-ERROR-NEXT:                       ^

        ldp x2, x5, [sp, #4]
        ldp x5, x6, [x9, #512]
        stp x7, x8, [x10, #-520]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp x2, x5, [sp, #4]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp x5, x6, [x9, #512]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stp x7, x8, [x10, #-520]
// CHECK-ERROR-NEXT:                     ^

        ldp sp, x3, [x10]
        stp x3, sp, [x9]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp sp, x3, [x10]
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stp x3, sp, [x9]
// CHECK-ERROR-NEXT:                 ^

        stp s3, s5, [sp, #-2]
        ldp s6, s26, [x4, #-260]
        stp s13, s19, [x5, #256]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp s3, s5, [sp, #-2]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp s6, s26, [x4, #-260]
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp s13, s19, [x5, #256]
// CHECK-ERROR-NEXT:                       ^

        ldp d3, d4, [xzr]
        ldp d5, d6, [x0, #512]
        stp d7, d8, [x0, #-520]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp d3, d4, [xzr]
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp d5, d6, [x0, #512]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stp d7, d8, [x0, #-520]
// CHECK-ERROR-NEXT:                     ^

        ldp d3, q2, [sp]
        ldp q3, q5, [sp, #8]
        stp q20, q25, [x5, #1024]
        ldp q30, q15, [x23, #-1040]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp d3, q2, [sp]
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldp q3, q5, [sp, #8]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         stp q20, q25, [x5, #1024]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldp q30, q15, [x23, #-1040]
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Load/store register pair (post-indexed)
//------------------------------------------------------------------------------

        ldp w3, w2, [x4], #1
        stp w1, w2, [x3], #253
        stp w9, w10, [x5], #256
        ldp w11, w12, [x9], #-260
        stp wsp, w9, [sp], #0
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp w3, w2, [x4], #1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp w1, w2, [x3], #253
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp w9, w10, [x5], #256
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp w11, w12, [x9], #-260
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stp wsp, w9, [sp], #0
// CHECK-ERROR-NEXT:             ^

        ldpsw x9, x2, [sp], #2
        ldpsw x1, x2, [x10], #256
        ldpsw x3, x4, [x11], #-260
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x9, x2, [sp], #2
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x1, x2, [x10], #256
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x3, x4, [x11], #-260
// CHECK-ERROR-NEXT:                       ^

        ldp x2, x5, [sp], #4
        ldp x5, x6, [x9], #512
        stp x7, x8, [x10], #-520
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp x2, x5, [sp], #4
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp x5, x6, [x9], #512
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stp x7, x8, [x10], #-520
// CHECK-ERROR-NEXT:                            ^

        ldp sp, x3, [x10], #0
        stp x3, sp, [x9], #0
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp sp, x3, [x10], #0
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stp x3, sp, [x9], #0
// CHECK-ERROR-NEXT:                 ^

        stp s3, s5, [sp], #-2
        ldp s6, s26, [x4], #-260
        stp s13, s19, [x5], #256
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp s3, s5, [sp], #-2
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp s6, s26, [x4], #-260
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp s13, s19, [x5], #256
// CHECK-ERROR-NEXT:                       ^

        ldp d3, d4, [xzr], #0
        ldp d5, d6, [x0], #512
        stp d7, d8, [x0], #-520
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp d3, d4, [xzr], #0
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp d5, d6, [x0], #512
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stp d7, d8, [x0], #-520
// CHECK-ERROR-NEXT:                     ^

        ldp d3, q2, [sp], #0
        ldp q3, q5, [sp], #8
        stp q20, q25, [x5], #1024
        ldp q30, q15, [x23], #-1040
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp d3, q2, [sp], #0
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldp q3, q5, [sp], #8
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         stp q20, q25, [x5], #1024
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldp q30, q15, [x23], #-1040
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Load/store register pair (pre-indexed)
//------------------------------------------------------------------------------

        ldp w3, w2, [x4, #1]!
        stp w1, w2, [x3, #253]!
        stp w9, w10, [x5, #256]!
        ldp w11, w12, [x9, #-260]!
        stp wsp, w9, [sp, #0]!
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp w3, w2, [x4, #1]!
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp w1, w2, [x3, #253]!
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp w9, w10, [x5, #256]!
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp w11, w12, [x9, #-260]!
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stp wsp, w9, [sp, #0]!
// CHECK-ERROR-NEXT:             ^

        ldpsw x9, x2, [sp, #2]!
        ldpsw x1, x2, [x10, #256]!
        ldpsw x3, x4, [x11, #-260]!
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x9, x2, [sp, #2]!
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x1, x2, [x10, #256]!
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldpsw x3, x4, [x11, #-260]!
// CHECK-ERROR-NEXT:                       ^

        ldp x2, x5, [sp, #4]!
        ldp x5, x6, [x9, #512]!
        stp x7, x8, [x10, #-520]!
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp x2, x5, [sp, #4]!
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp x5, x6, [x9, #512]!
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stp x7, x8, [x10, #-520]!
// CHECK-ERROR-NEXT:                     ^

        ldp sp, x3, [x10, #0]!
        stp x3, sp, [x9, #0]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp sp, x3, [x10, #0]!
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stp x3, sp, [x9, #0]!
// CHECK-ERROR-NEXT:                 ^

        stp s3, s5, [sp, #-2]!
        ldp s6, s26, [x4, #-260]!
        stp s13, s19, [x5, #256]!
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp s3, s5, [sp, #-2]!
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldp s6, s26, [x4, #-260]!
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stp s13, s19, [x5, #256]!
// CHECK-ERROR-NEXT:                       ^

        ldp d3, d4, [xzr, #0]!
        ldp d5, d6, [x0, #512]!
        stp d7, d8, [x0, #-520]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp d3, d4, [xzr, #0]!
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldp d5, d6, [x0, #512]!
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stp d7, d8, [x0, #-520]!
// CHECK-ERROR-NEXT:                     ^

        ldp d3, q2, [sp, #0]!
        ldp q3, q5, [sp, #8]!
        stp q20, q25, [x5, #1024]!
        ldp q30, q15, [x23, #-1040]!
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldp d3, q2, [sp, #0]!
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldp q3, q5, [sp, #8]!
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         stp q20, q25, [x5, #1024]!
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldp q30, q15, [x23, #-1040]!
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Load/store register pair (offset)
//------------------------------------------------------------------------------
        ldnp w3, w2, [x4, #1]
        stnp w1, w2, [x3, #253]
        stnp w9, w10, [x5, #256]
        ldnp w11, w12, [x9, #-260]
        stnp wsp, w9, [sp]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldnp w3, w2, [x4, #1]
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stnp w1, w2, [x3, #253]
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stnp w9, w10, [x5, #256]
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldnp w11, w12, [x9, #-260]
// CHECK-ERROR-NEXT:                             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stnp wsp, w9, [sp]
// CHECK-ERROR-NEXT:              ^

        ldnp x2, x5, [sp, #4]
        ldnp x5, x6, [x9, #512]
        stnp x7, x8, [x10, #-520]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldnp x2, x5, [sp, #4]
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldnp x5, x6, [x9, #512]
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stnp x7, x8, [x10, #-520]
// CHECK-ERROR-NEXT:                            ^

        ldnp sp, x3, [x10]
        stnp x3, sp, [x9]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldnp sp, x3, [x10]
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         stnp x3, sp, [x9]
// CHECK-ERROR-NEXT:                 ^

        stnp s3, s5, [sp, #-2]
        ldnp s6, s26, [x4, #-260]
        stnp s13, s19, [x5, #256]
// CHECK-ERROR: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stnp s3, s5, [sp, #-2]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         ldnp s6, s26, [x4, #-260]
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 4 in range [-256, 252]
// CHECK-ERROR-NEXT:         stnp s13, s19, [x5, #256]
// CHECK-ERROR-NEXT:                       ^

        ldnp d3, d4, [xzr]
        ldnp d5, d6, [x0, #512]
        stnp d7, d8, [x0, #-520]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldnp d3, d4, [xzr]
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         ldnp d5, d6, [x0, #512]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 8 in range [-512, 504]
// CHECK-ERROR-NEXT:         stnp d7, d8, [x0, #-520]
// CHECK-ERROR-NEXT:                     ^

        ldnp d3, q2, [sp]
        ldnp q3, q5, [sp, #8]
        stnp q20, q25, [x5, #1024]
        ldnp q30, q15, [x23, #-1040]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ldnp d3, q2, [sp]
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldnp q3, q5, [sp, #8]
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         stnp q20, q25, [x5, #1024]
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: {{expected integer|index must be a}} multiple of 16 in range [-1024, 1008]
// CHECK-ERROR-NEXT:         ldnp q30, q15, [x23, #-1040]
// CHECK-ERROR-NEXT:                       ^

//------------------------------------------------------------------------------
// Logical (shifted register)
//------------------------------------------------------------------------------
        orr w0, w1, #0xffffffff
        and x3, x5, #0xffffffffffffffff
// CHECK-ERROR: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         orr w0, w1, #0xffffffff
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         and x3, x5, #0xffffffffffffffff
// CHECK-ERROR-NEXT:                     ^

        ands w3, w9, #0x0
        eor x2, x0, #0x0
// CHECK-ERROR: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         ands w3, w9, #0x0
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         eor x2, x0, #0x0
// CHECK-ERROR-NEXT:                     ^

        eor w3, w5, #0x83
        eor x9, x20, #0x1234
// CHECK-ERROR: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         eor w3, w5, #0x83
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         eor x9, x20, #0x1234
// CHECK-ERROR-NEXT:                      ^

        and wzr, w4, 0xffff0000
        eor xzr, x9, #0xffff0000ffff0000
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         and wzr, w4, 0xffff0000
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         eor xzr, x9, #0xffff0000ffff0000
// CHECK-ERROR-NEXT:                      ^

        orr w3, wsp, #0xf0f0f0f0
        ands x3, sp, #0xaaaaaaaaaaaaaaaa
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         orr w3, wsp, #0xf0f0f0f0
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ands x3, sp, #0xaaaaaaaaaaaaaaaa
// CHECK-ERROR-NEXT:                  ^

        tst sp, #0xe0e0e0e0e0e0e0e0
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         tst sp, #0xe0e0e0e0e0e0e0e0
// CHECK-ERROR-NEXT:             ^

        // movi has been removed from the specification. Make sure it's really gone.
        movi wzr, #0x44444444
        movi w3, #0xffff
        movi x9, #0x0000ffff00000000
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         movi wzr, #0x44444444
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         movi w3, #0xffff
// CHECK-ERROR-NEXT:         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         movi x9, #0x0000ffff00000000
// CHECK-ERROR-NEXT:         ^

//------------------------------------------------------------------------------
// Logical (shifted register)
//------------------------------------------------------------------------------

        //// Out of range shifts
        and w2, w24, w6, lsl #-1
        and w4, w6, w12, lsl #32
        and x4, x6, x12, lsl #64
        and x2, x5, x11, asr
// CHECK-ERROR: error: expected integer shift amount
// CHECK-ERROR-NEXT:         and w2, w24, w6, lsl #-1
// CHECK-ERROR-NEXT:                               ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]
// CHECK-ERROR-NEXT:         and w4, w6, w12, lsl #32
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]
// CHECK-ERROR-NEXT:         and x4, x6, x12, lsl #64
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: expected #imm after shift specifier
// CHECK-ERROR-NEXT:         and x2, x5, x11, asr
// CHECK-ERROR-NEXT:                             ^

        //// sp not allowed
        orn wsp, w3, w5
        bics x20, sp, x9, lsr #0
        orn x2, x6, sp, lsl #3
// FIXME: the diagnostic we get for 'orn wsp, w3, w5' is from the orn alias,
// which is a better match than the genuine ORNWri, whereas it would be better
// to get the ORNWri diagnostic when the alias did not match, i.e. the
// alias' diagnostics should have a lower priority.
// CHECK-ERROR: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         orn wsp, w3, w5
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         bics x20, sp, x9, lsr #0
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         orn x2, x6, sp, lsl #3
// CHECK-ERROR-NEXT:                     ^

        //// Mismatched registers
        and x3, w2, w1
        ands w1, x12, w2
        and x4, x5, w6, lsl #12
        orr w2, w5, x7, asr #0
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         and x3, w2, w1
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         ands w1, x12, w2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         and x4, x5, w6, lsl #12
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: expected compatible register or logical immediate
// CHECK-ERROR-NEXT:         orr w2, w5, x7, asr #0
// CHECK-ERROR-NEXT:                     ^

        //// Shifts should not be allowed on mov
        mov w3, w7, lsl #13
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         mov w3, w7, lsl #13
// CHECK-ERROR-NEXT:                     ^

//------------------------------------------------------------------------------
// Move wide (immediate)
//------------------------------------------------------------------------------

        movz w3, #65536, lsl #0
        movz w4, #65536
        movn w1, #2, lsl #1
        movk w3, #0, lsl #-1
        movn w2, #-1, lsl #0
        movz x3, #-1
        movk w3, #1, lsl #32
        movn x2, #12, lsl #64
// CHECK-ERROR: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz w3, #65536, lsl #0
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz w4, #65536
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: expected relocated symbol or integer in range [0, 65535]
// CHECK-ERROR-ARM64-NEXT: error: expected 'lsl' with optional integer 0 or 16
// CHECK-ERROR-NEXT:         movn w1, #2, lsl #1
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: only 'lsl #+N' valid after immediate
// CHECK-ERROR-ARM64-NEXT: error: expected integer shift amount
// CHECK-ERROR-NEXT:         movk w3, #0, lsl #-1
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movn w2, #-1, lsl #0
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz x3, #-1
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: expected relocated symbol or integer in range [0, 65535]
// CHECK-ERROR-ARM64-NEXT: error: expected 'lsl' with optional integer 0 or 16
// CHECK-ERROR-NEXT:         movk w3, #1, lsl #32
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-AARCH64-NEXT: error: expected relocated symbol or integer in range [0, 65535]
// CHECK-ERROR-ARM64-NEXT: error: expected 'lsl' with optional integer 0, 16, 32 or 48
// CHECK-ERROR-NEXT:         movn x2, #12, lsl #64
// CHECK-ERROR-NEXT:                  ^

        movz x12, #:abs_g0:sym, lsl #16
        movz x12, #:abs_g0:sym, lsl #0
        movn x2, #:abs_g0:sym
        movk w3, #:abs_g0:sym
        movz x3, #:abs_g0_nc:sym
        movn x4, #:abs_g0_nc:sym
// CHECK-ERROR: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz x12, #:abs_g0:sym, lsl #16
// CHECK-ERROR-NEXT:                                 ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz x12, #:abs_g0:sym, lsl #0
// CHECK-ERROR-NEXT:                                 ^
// CHECK-ERROR-AARCH64-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-AARCH64-NEXT:         movn x2, #:abs_g0:sym
// CHECK-ERROR-AARCH64-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w3, #:abs_g0:sym
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz x3, #:abs_g0_nc:sym
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movn x4, #:abs_g0_nc:sym
// CHECK-ERROR-NEXT:                  ^

        movn x2, #:abs_g1:sym
        movk w3, #:abs_g1:sym
        movz x3, #:abs_g1_nc:sym
        movn x4, #:abs_g1_nc:sym
// CHECK-ERROR-AARCH64: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-AARCH64-NEXT:         movn x2, #:abs_g1:sym
// CHECK-ERROR-AARCH64-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w3, #:abs_g1:sym
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz x3, #:abs_g1_nc:sym
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movn x4, #:abs_g1_nc:sym
// CHECK-ERROR-NEXT:                  ^

        movz w12, #:abs_g2:sym
        movn x12, #:abs_g2:sym
        movk x13, #:abs_g2:sym
        movk w3, #:abs_g2_nc:sym
        movz x13, #:abs_g2_nc:sym
        movn x24, #:abs_g2_nc:sym
// CHECK-ERROR: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz w12, #:abs_g2:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-AARCH64-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-AARCH64-NEXT:         movn x12, #:abs_g2:sym
// CHECK-ERROR-AARCH64-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk x13, #:abs_g2:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w3, #:abs_g2_nc:sym
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz x13, #:abs_g2_nc:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movn x24, #:abs_g2_nc:sym
// CHECK-ERROR-NEXT:                   ^

        movn x19, #:abs_g3:sym
        movz w20, #:abs_g3:sym
        movk w21, #:abs_g3:sym
// CHECK-ERROR-AARCH64: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-AARCH64-NEXT:         movn x19, #:abs_g3:sym
// CHECK-ERROR-AARCH64-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz w20, #:abs_g3:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w21, #:abs_g3:sym
// CHECK-ERROR-NEXT:                   ^

        movk x19, #:abs_g0_s:sym
        movk w23, #:abs_g0_s:sym
// CHECK-ERROR: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk x19, #:abs_g0_s:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w23, #:abs_g0_s:sym
// CHECK-ERROR-NEXT:                   ^

        movk x19, #:abs_g1_s:sym
        movk w23, #:abs_g1_s:sym
// CHECK-ERROR: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk x19, #:abs_g1_s:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w23, #:abs_g1_s:sym
// CHECK-ERROR-NEXT:                   ^

        movz w2, #:abs_g2_s:sym
        movn w29, #:abs_g2_s:sym
        movk x19, #:abs_g2_s:sym
        movk w23, #:abs_g2_s:sym
// CHECK-ERROR: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movz w2, #:abs_g2_s:sym
// CHECK-ERROR-NEXT:                    ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movn w29, #:abs_g2_s:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk x19, #:abs_g2_s:sym
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: {{expected relocated symbol or|immediate must be an}} integer in range [0, 65535]
// CHECK-ERROR-NEXT:         movk w23, #:abs_g2_s:sym
// CHECK-ERROR-NEXT:                   ^

//------------------------------------------------------------------------------
// PC-relative addressing
//------------------------------------------------------------------------------

        adr sp, loc             // expects xzr
        adrp x3, #20            // Immediate unaligned
        adrp w2, loc            // 64-bit register needed
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adr sp, loc
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         adrp x3, #20
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         adrp w2, loc
// CHECK-ERROR-NEXT:              ^

        adr x9, #1048576
        adr x2, #-1048577
        adrp x9, #4294967296
        adrp x20, #-4294971392
// CHECK-ERROR: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         adr x9, #1048576
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         adr x2, #-1048577
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         adrp x9, #4294967296
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         adrp x20, #-4294971392
// CHECK-ERROR-NEXT:                   ^

//------------------------------------------------------------------------------
// System
//------------------------------------------------------------------------------

        hint #-1
        hint #128
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 127]
// CHECK-ERROR-NEXT:         hint #-1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 127]
// CHECK-ERROR-NEXT:         hint #128
// CHECK-ERROR-NEXT:              ^

        clrex #-1
        clrex #16
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:         clrex #-1
// CHECK-ERROR-NEXT:               ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR-NEXT:         clrex #16
// CHECK-ERROR-NEXT:               ^

        dsb #-1
        dsb #16
        dsb foo
        dmb #-1
        dmb #16
        dmb foo
// CHECK-ERROR-NEXT: error: {{Invalid immediate for instruction|barrier operand out of range}}
// CHECK-ERROR-NEXT:         dsb #-1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{Invalid immediate for instruction|barrier operand out of range}}
// CHECK-ERROR-NEXT:         dsb #16
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid barrier option name
// CHECK-ERROR-NEXT:         dsb foo
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{Invalid immediate for instruction|barrier operand out of range}}
// CHECK-ERROR-NEXT:         dmb #-1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{Invalid immediate for instruction|barrier operand out of range}}
// CHECK-ERROR-NEXT:         dmb #16
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: invalid barrier option name
// CHECK-ERROR-NEXT:         dmb foo
// CHECK-ERROR-NEXT:             ^

        isb #-1
        isb #16
        isb foo
// CHECK-ERROR-NEXT: error: {{Invalid immediate for instruction|barrier operand out of range}}
// CHECK-ERROR-NEXT:         isb #-1
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{Invalid immediate for instruction|barrier operand out of range}}
// CHECK-ERROR-NEXT:         isb #16
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: 'sy' or #imm operand expected
// CHECK-ERROR-NEXT:        isb foo
// CHECK-ERROR-NEXT:            ^

        msr daifset, x4
        msr spsel, #-1
        msr spsel #-1
        msr daifclr, #16
// CHECK-ERROR: [[@LINE-4]]:22: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR: [[@LINE-4]]:20: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR: [[@LINE-4]]:{{9|19}}: error: {{too few operands for instruction|expected comma before next operand|unexpected token in argument list}}
// CHECK-ERROR: [[@LINE-4]]:22: error: {{expected|immediate must be an}} integer in range [0, 15]

        sys #8, c1, c2, #7, x9
        sys #3, c16, c2, #3, x10
        sys #2, c11, c16, #5
        sys #4, c9, c8, #8, xzr
        sysl x11, #8, c1, c2, #7
        sysl x13, #3, c16, c2, #3
        sysl x9, #2, c11, c16, #5
        sysl x4, #4, c9, c8, #8
// CHECK-ERROR: error:  {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR-NEXT:         sys #8, c1, c2, #7, x9
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: Expected cN operand where 0 <= N <= 15
// CHECK-ERROR-NEXT:         sys #3, c16, c2, #3, x10
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: Expected cN operand where 0 <= N <= 15
// CHECK-ERROR-NEXT:         sys #2, c11, c16, #5
// CHECK-ERROR-NEXT:                      ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR-NEXT:         sys #4, c9, c8, #8, xzr
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR-NEXT:         sysl x11, #8, c1, c2, #7
// CHECK-ERROR-NEXT:                   ^
// CHECK-ERROR-NEXT: error: Expected cN operand where 0 <= N <= 15
// CHECK-ERROR-NEXT:         sysl x13, #3, c16, c2, #3
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: Expected cN operand where 0 <= N <= 15
// CHECK-ERROR-NEXT:         sysl x9, #2, c11, c16, #5
// CHECK-ERROR-NEXT:                           ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR-NEXT:         sysl x4, #4, c9, c8, #8
// CHECK-ERROR-NEXT:                              ^

        ic ialluis, x2
        ic allu, x7
        ic ivau
// CHECK-ERROR-NEXT: error: specified {{IC|ic}} op does not use a register
// CHECK-ERROR-NEXT:         ic ialluis, x2
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-AARCH64-NEXT: error: operand specifier not recognised
// CHECK-ERROR-ARM64-NEXT: error: invalid operand for IC instruction
// CHECK-ERROR-NEXT:         ic allu, x7
// CHECK-ERROR-NEXT:            ^
// CHECK-ERROR-NEXT: error: specified {{IC|ic}} op requires a register
// CHECK-ERROR-NEXT:         ic ivau
// CHECK-ERROR-NEXT:            ^

        tlbi IPAS2E1IS
        tlbi IPAS2LE1IS
        tlbi VMALLE1IS, x12
        tlbi ALLE2IS, x11
        tlbi ALLE3IS, x20
        tlbi VAE1IS
        tlbi VAE2IS
        tlbi VAE3IS
        tlbi ASIDE1IS
        tlbi VAAE1IS
        tlbi ALLE1IS, x0
        tlbi VALE1IS
        tlbi VALE2IS
        tlbi VALE3IS
        tlbi VMALLS12E1IS, xzr
        tlbi VAALE1IS
        tlbi IPAS2E1
        tlbi IPAS2LE1
        tlbi VMALLE1, x9
        tlbi ALLE2, x10
        tlbi ALLE3, x11
        tlbi VAE1
        tlbi VAE2
        tlbi VAE3
        tlbi ASIDE1
        tlbi VAAE1
        tlbi ALLE1, x25
        tlbi VALE1
        tlbi VALE2
        tlbi VALE3
        tlbi VMALLS12E1, x15
        tlbi VAALE1
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi IPAS2E1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi IPAS2LE1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi VMALLE1IS, x12
// CHECK-ERROR-NEXT:                         ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi ALLE2IS, x11
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi ALLE3IS, x20
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAE1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAE2IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAE3IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi ASIDE1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAAE1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi ALLE1IS, x0
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VALE1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VALE2IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VALE3IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi VMALLS12E1IS, xzr
// CHECK-ERROR-NEXT:                            ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAALE1IS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi IPAS2E1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi IPAS2LE1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi VMALLE1, x9
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi ALLE2, x10
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi ALLE3, x11
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAE1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAE2
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAE3
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi ASIDE1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAAE1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi ALLE1, x25
// CHECK-ERROR-NEXT:                     ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VALE1
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VALE2
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VALE3
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi VMALLS12E1, x15
// CHECK-ERROR-NEXT:                          ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi VAALE1
// CHECK-ERROR-NEXT:              ^

// For the MSR/MRS instructions, first make sure read-only and
// write-only registers actually are.
        msr MDCCSR_EL0, x12
        msr DBGDTRRX_EL0, x12
        msr MDRAR_EL1, x12
        msr OSLSR_EL1, x12
        msr DBGAUTHSTATUS_EL1, x12
        msr MIDR_EL1, x12
        msr CCSIDR_EL1, x12
        msr CLIDR_EL1, x12
        msr CTR_EL0, x12
        msr MPIDR_EL1, x12
        msr REVIDR_EL1, x12
        msr AIDR_EL1, x12
        msr DCZID_EL0, x12
        msr ID_PFR0_EL1, x12
        msr ID_PFR1_EL1, x12
        msr ID_DFR0_EL1, x12
        msr ID_AFR0_EL1, x12
        msr ID_MMFR0_EL1, x12
        msr ID_MMFR1_EL1, x12
        msr ID_MMFR2_EL1, x12
        msr ID_MMFR3_EL1, x12
        msr ID_MMFR4_EL1, x12
        msr ID_ISAR0_EL1, x12
        msr ID_ISAR1_EL1, x12
        msr ID_ISAR2_EL1, x12
        msr ID_ISAR3_EL1, x12
        msr ID_ISAR4_EL1, x12
        msr ID_ISAR5_EL1, x12
        msr MVFR0_EL1, x12
        msr MVFR1_EL1, x12
        msr MVFR2_EL1, x12
        msr ID_AA64PFR0_EL1, x12
        msr ID_AA64PFR1_EL1, x12
        msr ID_AA64DFR0_EL1, x12
        msr ID_AA64DFR1_EL1, x12
        msr ID_AA64AFR0_EL1, x12
        msr ID_AA64AFR1_EL1, x12
        msr ID_AA64ISAR0_EL1, x12
        msr ID_AA64ISAR1_EL1, x12
        msr ID_AA64MMFR0_EL1, x12
        msr ID_AA64MMFR1_EL1, x12
        msr PMCEID0_EL0, x12
        msr PMCEID1_EL0, x12
        msr RVBAR_EL1, x12
        msr RVBAR_EL2, x12
        msr RVBAR_EL3, x12
        msr ISR_EL1, x12
        msr CNTPCT_EL0, x12
        msr CNTVCT_EL0, x12
        msr PMEVCNTR31_EL0, x12
        msr PMEVTYPER31_EL0, x12
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MDCCSR_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr DBGDTRRX_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MDRAR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr OSLSR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr DBGAUTHSTATUS_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MIDR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr CCSIDR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr CLIDR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr CTR_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MPIDR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr REVIDR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr AIDR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr DCZID_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_PFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_PFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_DFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_MMFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_MMFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_MMFR2_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_MMFR3_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_MMFR4_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_ISAR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_ISAR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_ISAR2_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_ISAR3_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_ISAR4_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_ISAR5_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MVFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MVFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr MVFR2_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64PFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64PFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64DFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64DFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64AFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64AFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64ISAR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64ISAR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64MMFR0_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ID_AA64MMFR1_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr PMCEID0_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr PMCEID1_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr RVBAR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr RVBAR_EL2, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr RVBAR_EL3, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr ISR_EL1, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr CNTPCT_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr CNTVCT_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr PMEVCNTR31_EL0, x12
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr PMEVTYPER31_EL0, x12
// CHECK-ERROR-NEXT:             ^

        mrs x9, DBGDTRTX_EL0
        mrs x9, OSLAR_EL1
        mrs x9, PMSWINC_EL0
        mrs x9, PMEVCNTR31_EL0
        mrs x9, PMEVTYPER31_EL0
// CHECK-ERROR: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x9, DBGDTRTX_EL0
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x9, OSLAR_EL1
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x9, PMSWINC_EL0
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x9, PMEVCNTR31_EL0
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x9, PMEVTYPER31_EL0
// CHECK-ERROR-NEXT:                 ^

// Now check some invalid generic names
        mrs x12, s3_8_c11_c13_2
        mrs x19, s3_2_c15_c16_2
        mrs x30, s3_2_c15_c1_8
        mrs x4, s4_7_c15_c15_7
        mrs x14, s3_7_c16_c15_7
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x12, s3_8_c11_c13_2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x19, s3_2_c15_c16_2
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x30, s3_2_c15_c1_8
// CHECK-ERROR-NEXT:                  ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x4, s4_7_c15_c15_7
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x14, s3_7_c16_c15_7
// CHECK-ERROR-NEXT:                  ^

//------------------------------------------------------------------------------
// Test and branch (immediate)
//------------------------------------------------------------------------------

        tbz w3, #-1, addr
        tbz w3, #32, nowhere
        tbz x9, #-1, there
        tbz x20, #64, dont
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:     tbz w3, #-1, addr
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        tbz w3, #32, nowhere
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:        tbz x9, #-1, there
// CHECK-ERROR-NEXT:                ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:        tbz x20, #64, dont
// CHECK-ERROR-NEXT:                 ^

        tbnz w3, #-1, addr
        tbnz w3, #32, nowhere
        tbnz x9, #-1, there
        tbnz x20, #64, dont
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        tbnz w3, #-1, addr
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR-NEXT:        tbnz w3, #32, nowhere
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:        tbnz x9, #-1, there
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR-NEXT:        tbnz x20, #64, dont

//------------------------------------------------------------------------------
// Unconditional branch (immediate)
//------------------------------------------------------------------------------

        b #134217728
        b #-134217732
        b #1
// CHECK-ERROR: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         b #134217728
// CHECK-ERROR-NEXT:           ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         b #-134217732
// CHECK-ERROR-NEXT:           ^
// CHECK-ERROR-NEXT: error: expected label or encodable integer pc offset
// CHECK-ERROR-NEXT:         b #1
// CHECK-ERROR-NEXT:           ^

//------------------------------------------------------------------------------
// Unconditional branch (register)
//------------------------------------------------------------------------------

        br w2
        br sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         br w2
// CHECK-ERROR-NEXT:            ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         br sp
// CHECK-ERROR-NEXT:            ^

        //// These ones shouldn't allow any registers
        eret x2
        drps x2
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         eret x2
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: invalid operand for instruction
// CHECK-ERROR-NEXT:         drps x2
// CHECK-ERROR-NEXT:              ^

