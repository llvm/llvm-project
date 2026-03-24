// +tbl-rmi required for RIPA*/RVA*
// +xs required for *NXS

// RUN: not llvm-mc -triple aarch64 -mattr=+d128,+tlb-rmi,+xs -show-encoding %s -o - 2>&1 | FileCheck %s --check-prefix=ERRORS

// sysp #<op1>, <Cn>, <Cm>, #<op2>{, <Xt1>, <Xt2>}
// registers with 128-bit formats (op0, op1, Cn, Cm, op2)
// For sysp, op0 is 0

sysp #0, c8, c0, #0, x0, x2
// ERRORS: error: expected second odd register of a consecutive 64-bit register pair
sysp #4, c8, c4, #1, x1, x2
// ERRORS: error: expected xzr/xzr or the first even register of a consecutive 64-bit register pair
sysp #4, c8, c4, #1, x2, x4
// ERRORS: error: expected second odd register of a consecutive 64-bit register pair
sysp #0, c8, c0, #0, x29, x31
// ERRORS: error: expected xzr/xzr or the first even register of a consecutive 64-bit register pair
sysp #0, c8, c0, #0, x30, x30
// ERRORS: error: expected second odd register of a consecutive 64-bit register pair
sysp #0, c8, c0, #0, x31, x0
// ERRORS: error: expected second xzr in xzr/xzr register pair
sysp #4, c8, c4, #1, xzr, x1
// ERRORS: error: expected second xzr in xzr/xzr register pair
sysp #0, c8, c0, #0, xzr, x30
// ERRORS: error: expected second xzr in xzr/xzr register pair
sysp #0, c8, c0, #0, w0, w1
// ERRORS: error: expected xzr/xzr or the first even register of a consecutive 64-bit register pair
sysp #7, c8, c0, #0, x0, x1
// ERRORS: error: immediate must be an integer in range [0, 6].
sysp #0, c7, c0, #0, x0, x1
// ERRORS: error: expected cN operand where 8 <= N <= 9
sysp #0, c10, c0, #0, x0, x1
// ERRORS: error: expected cN operand where 8 <= N <= 9
sysp #0, c8, c8, #0, x0, x1
// ERRORS: error: expected cN operand where 0 <= N <= 7
sysp #0, c8, c0, #8, x0, x1
// ERRORS: error: immediate must be an integer in range [0, 7].
sysp #0, c8, c0, #0, xzr,
// ERRORS: error: expected second xzr in xzr/xzr register pair

tlbip RVAE3IS
// ERRORS: error: expected comma
tlbip RVAE3IS,
// ERRORS: error: expected register identifier
tlbip VAE3,
// ERRORS: error: expected register identifier
tlbip IPAS2E1, x4, x8
// ERRORS: error: expected second odd register of a consecutive 64-bit register pair
tlbip RVAE3, x11, x11
// ERRORS: error: expected xzr/xzr or the first even register of a consecutive 64-bit register pair

sysp #0, c8, c0, #0, x0
// ERRORS: error: expected comma
sysp #0, c8, c0, #0, xzr
// ERRORS: error: expected comma
