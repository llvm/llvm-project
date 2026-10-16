// +tbl-rmi required for RIPA*/RVA*
// +xs required for *NXS

// RUN: not llvm-mc -triple aarch64 -mattr=+d128,+tlb-rmi,+xs -show-encoding %s -o - 2>&1 | FileCheck %s --check-prefix=ERRORS

// sysp #<op1>, <Cn>, <Cm>, #<op2>{, <Xt1>, <Xt2>}
// registers with 128-bit formats (op0, op1, Cn, Cm, op2)
// For sysp, op0 is 0

sysp #0, c2, c0, #0, x0, x2
// ERRORS: error: expected second odd register of a consecutive same-size even/odd register pair
sysp #0, c2, c0, #0, x0
// ERRORS: error: expected comma
sysp #0, c2, c0, #0, x1, x2
// ERRORS: error: expected first even register of a consecutive same-size even/odd register pair
sysp #0, c2, c0, #0, x31, x0
// ERRORS: error: xzr must be followed by xzr
sysp #0, c2, c0, #0, xzr, x30
// ERRORS: error: xzr must be followed by xzr
sysp #0, c2, c0, #0, xzr
// ERRORS: error: expected comma
sysp #0, c2, c0, #0, xzr,
// ERRORS: error: expected register operand


tlbip RVAE3IS
// ERRORS: error: expected comma
tlbip RVAE3IS,
// ERRORS: error: expected register identifier
tlbip VAE3,
// ERRORS: error: expected register identifier
tlbip IPAS2E1, x4, x8
// ERRORS: error: specified tlbip op requires a pair of registers
tlbip RVAE3, x11, x11
// ERRORS: error: specified tlbip op requires a pair of registers
