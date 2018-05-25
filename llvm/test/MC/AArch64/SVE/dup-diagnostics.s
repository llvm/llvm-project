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

dup z0.b, #-129
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 255]
// CHECK-NEXT: dup z0.b, #-129
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

dup z0.h, #-33024
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.h, #-33024
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #-32769
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.h, #-32769
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #-129, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.h, #-129, lsl #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #32513
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.h, #32513
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #32768
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.h, #32768
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

dup z0.h, #128, lsl #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [-128, 127] or a multiple of 256 in range [-32768, 32512]
// CHECK-NEXT: dup z0.h, #128, lsl #8
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
