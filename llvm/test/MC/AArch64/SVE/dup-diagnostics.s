// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// input should be a 64bit scalar register
dup z0.d, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.d, w0
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// wzr is not a valid operand to dup
dup z0.s, wzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.s, wzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// xzr is not a valid operand to dup
dup z0.d, xzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.d, xzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

dup z0.b, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: dup z0.b, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.b, #-1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: dup z0.b, #-1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.b, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: dup z0.b, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.b, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: dup z0.b, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: dup z0.h, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #65281
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: dup z0.h, #65281
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #65536
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: dup z0.h, #65536
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: dup z0.h, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.s, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.s, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.s, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.s, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.s, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.s, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.d, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.d, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.d, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.d, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.d, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.d, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not compatible with encode/decode function.

dup z0.b, b0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.b, b0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, h0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.h, h0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.s, s0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.s, s0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.d, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.d, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.q, q0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: dup z0.q, q0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z24.b, z17.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 63].
// CHECK-NEXT: dup z24.b, z17.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z17.b, z5.b[64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 63].
// CHECK-NEXT: dup z17.b, z5.b[64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z16.h, z30.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 31].
// CHECK-NEXT: dup z16.h, z30.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z19.h, z23.h[32]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 31].
// CHECK-NEXT: dup z19.h, z23.h[32]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z1.s, z6.s[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: dup z1.s, z6.s[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z24.s, z3.s[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: dup z24.s, z3.s[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z5.d, z25.d[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: dup z5.d, z25.d[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z12.d, z28.d[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: dup z12.d, z28.d[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z22.q, z7.q[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: dup z22.q, z7.q[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z24.q, z21.q[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: dup z24.q, z21.q[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.b, p0/z, z6.b
dup     z31.b, wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: dup     z31.b, wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
dup     z31.b, wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: dup     z31.b, wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21.d, p0/z, z28.d
dup     z21.d, #32512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: dup     z21.d, #32512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
dup     z21.d, #32512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: dup     z21.d, #32512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p0/z, z6.d
dup     z31.d, z31.d[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: dup     z31.d, z31.d[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
dup     z31.d, z31.d[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: dup     z31.d, z31.d[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
