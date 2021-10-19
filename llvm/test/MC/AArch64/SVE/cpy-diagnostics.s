// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid scalar operand for result element width.

cpy z0.b, p0/m, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.b, p0/m, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/m, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.h, p0/m, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/m, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.s, p0/m, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/m, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.d, p0/m, w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.b, p0/m, h0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.b, p0/m, h0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.b, p0/m, s0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.b, p0/m, s0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.b, p0/m, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.b, p0/m, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/m, b0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.h, p0/m, b0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/m, s0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.h, p0/m, s0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/m, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.h, p0/m, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/m, b0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.s, p0/m, b0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/m, h0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.s, p0/m, h0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/m, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.s, p0/m, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/m, b0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.d, p0/m, b0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/m, h0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.d, p0/m, h0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/m, s0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cpy z0.d, p0/m, s0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

cpy z0.b, p0/z, #0, lsl #8      // #0, lsl #8 is not valid for .b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: cpy z0.b, p0/z, #0, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.b, p0/z, #-1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: cpy z0.b, p0/z, #-1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.b, p0/z, #256
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: cpy z0.b, p0/z, #256
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.b, p0/z, #1, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: cpy z0.b, p0/z, #1, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/z, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: cpy z0.h, p0/z, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/z, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: cpy z0.h, p0/z, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/z, #65281
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: cpy z0.h, p0/z, #65281
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.h, p0/z, #256, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 65280]
// CHECK-NEXT: cpy z0.h, p0/z, #256, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/z, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.s, p0/z, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/z, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.s, p0/z, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/z, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.s, p0/z, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/z, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.s, p0/z, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/z, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.s, p0/z, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.s, p0/z, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.s, p0/z, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/z, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.d, p0/z, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/z, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.d, p0/z, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/z, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.d, p0/z, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/z, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.d, p0/z, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/z, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.d, p0/z, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cpy z0.d, p0/z, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: cpy z0.d, p0/z, #128, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
