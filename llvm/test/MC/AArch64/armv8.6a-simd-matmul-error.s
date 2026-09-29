// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+i8mm       < %s 2>&1 | FileCheck %s

// No interesting edge cases for [US]MMLA, except for the fact that the data
// types are fixed (no 64-bit version), and USMMLA exists, but SUMMLA does not.
smmla  v1.2s, v16.8b, v31.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
summla v1.4s, v16.16b, v31.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unrecognized instruction mnemonic, did you mean: smmla, ummla, usmmla?

// USDOT (vector) has two valid data type combinations, others are rejected.
usdot v3.4s, v15.8b, v30.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
usdot v3.2s, v15.16b, v30.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// For USDOT and SUDOT (indexed), the index is in range [0,3] (regardless of data types)
usdot v31.2s, v1.8b,  v2.4b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
usdot v31.4s, v1.16b, v2.4b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
sudot v31.2s, v1.8b,  v2.4b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
sudot v31.4s, v1.16b, v2.4b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].

// The arrangement specifiers of the first two operands must match.
usdot v31.4s, v1.8b,  v2.4b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
usdot v31.2s, v1.16b, v2.4b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
sudot v31.4s, v1.8b,  v2.4b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
sudot v31.2s, v1.16b, v2.4b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
