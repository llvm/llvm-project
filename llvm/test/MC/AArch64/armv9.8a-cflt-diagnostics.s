// RUN: not llvm-mc -triple=aarch64 -mattr=+cflt -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s

flt.me #0
// CHECK: [[@LINE-1]]:5: error: invalid condition code

flt.eq #65536
// CHECK: [[@LINE-1]]:8: error: immediate must be an integer in range [0, 65535].

cflteq #0, wsp, #0
// CHECK: [[@LINE-1]]:17: error: invalid operand for instruction

cflteq #0, sp, #0
// CHECK: [[@LINE-1]]:16: error: invalid operand for instruction

cflteq #4, w0, #1
// CHECK: [[@LINE-1]]:8: error: immediate must be an integer in range [0, 3].

cfltgt #0, w0, #-257
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-256, 255].

cfltgt #0, w0, #256
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-256, 255].

cflthi #0, w0, #-1
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [0, 511].

cflthi #0, w0, #512
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [0, 511].

cfltge #0, w0, #-256
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-255, 256].

cfltge #0, w0, #257
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-255, 256].

cflths #0, w0, #0
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [1, 512].

cflths #0, w0, #513
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [1, 512].

cfltle #0, w0, #-258
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-257, 254].

cfltle #0, w0, #255
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-257, 254].

cfltls #0, w0, #-2
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-1, 510].

cfltls #0, w0, #511
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [-1, 510].

cfltlo #0, w0, #-1
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [0, 511].

cfltlo #0, w0, #512
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [0, 511].

cflteq #0, wzr, w0
// CHECK: [[@LINE-1]]:17: error: invalid operand for instruction

cflteq #0, xzr, x0
// CHECK: [[@LINE-1]]:17: error: invalid operand for instruction

cflteq #0, w0, wzr
// CHECK: [[@LINE-1]]:16: error: invalid operand for instruction

cflteq #0, x0, xzr
// CHECK: [[@LINE-1]]:16: error: invalid operand for instruction

cflteq #0, w0, label
// CHECK: [[@LINE-1]]:16: error: invalid operand for instruction

cflthi #0, w0, wzr
// CHECK: [[@LINE-1]]:16: error: invalid operand for instruction

cflths #0, x0, xzr
// CHECK: [[@LINE-1]]:16: error: invalid operand for instruction

cfltz #0, wsp
// CHECK: [[@LINE-1]]:11: error: invalid operand for instruction

cfltnz #0, sp
// CHECK: [[@LINE-1]]:12: error: invalid operand for instruction

tfltz #4, w0, #0
// CHECK: [[@LINE-1]]:7: error: immediate must be an integer in range [0, 3].

tfltz #0, w0, #32
// CHECK: [[@LINE-1]]:15: error: immediate must be an integer in range [0, 31].

tfltz #0, w0, #-1
// CHECK: [[@LINE-1]]:15: error: immediate must be an integer in range [0, 31].

tfltnz #4, x0, #32
// CHECK: [[@LINE-1]]:8: error: immediate must be an integer in range [0, 3].

tfltnz #0, x0, #64
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [0, 63].

tfltnz #0, x0, #-1
// CHECK: [[@LINE-1]]:16: error: immediate must be an integer in range [0, 63].
