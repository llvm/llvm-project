// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// input should be a 64bit scalar register
mov z0.d, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mov z0.d, w0
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// wzr is not a valid operand to mov
mov z0.s, wzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mov z0.s, wzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// xzr is not a valid operand to mov
mov z0.d, xzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mov z0.d, xzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Unpredicated mov of Z register only allowed for .d

mov z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK-NEXT: mov z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK-NEXT: mov z0.h, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK-NEXT: mov z0.s, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediates

mov z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, #-129
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, #-129
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, #-1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, #-1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Note: 65281 is a valid logical immediate.
mov z0.h, #65282
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, #65282
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z5.b, #0xfff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255] with a shift amount of 0
// CHECK-NEXT: mov z5.b, #0xfff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z5.h, #0xfffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z5.h, #0xfffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z5.h, #0xfffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z5.h, #0xfffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z5.s, #0xfffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z5.s, #0xfffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z5.s, #0xffffffffffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z5.s, #0xffffffffffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, p0/z, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, p0/z, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, p0/z, #-129
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, p0/z, #-129
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, p0/z, #-1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, p0/z, #-1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, p0/z, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, p0/z, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.b, p0/z, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: mov z0.b, p0/z, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, p0/z, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, p0/z, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, p0/z, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, p0/z, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, p0/z, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, p0/z, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, p0/z, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, p0/z, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, p0/z, #65281
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, p0/z, #65281
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.h, p0/z, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: mov z0.h, p0/z, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, p0/z, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, p0/z, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, p0/z, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, p0/z, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, p0/z, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, p0/z, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, p0/z, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, p0/z, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, p0/z, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, p0/z, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.s, p0/z, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.s, p0/z, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, p0/z, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, p0/z, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, p0/z, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, p0/z, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, p0/z, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, p0/z, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, p0/z, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, p0/z, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, p0/z, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, p0/z, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z0.d, p0/z, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: mov z0.d, p0/z, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not compatible with encode/decode function.

mov z24.b, z17.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 63].
// CHECK-NEXT: mov z24.b, z17.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z17.b, z5.b[64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 63].
// CHECK-NEXT: mov z17.b, z5.b[64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z16.h, z30.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 31].
// CHECK-NEXT: mov z16.h, z30.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z19.h, z23.h[32]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 31].
// CHECK-NEXT: mov z19.h, z23.h[32]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z1.s, z6.s[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: mov z1.s, z6.s[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z24.s, z3.s[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: mov z24.s, z3.s[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z5.d, z25.d[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: mov z5.d, z25.d[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z12.d, z28.d[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: mov z12.d, z28.d[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z22.q, z7.q[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: mov z22.q, z7.q[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

mov z24.q, z21.q[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: mov z24.q, z21.q[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
