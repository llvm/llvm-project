// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+8msecext -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK %s
// RUN:   FileCheck --check-prefix=ERROR < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+8msecext -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK %s
// RUN:   FileCheck --check-prefix=ERROR < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-none-eabi -mattr=-8msecext < %s 2>%t
// RUN: FileCheck --check-prefix=NOSEC < %t %s

// CHECK: vscclrm            {s0, s1, s2, s3, vpr} @ encoding: [0x9f,0xec,0x04,0x0a]
// NOSEC: instruction requires: ARMv8-M Security Extensions
vscclrm {s0-s3, vpr}

// CHECK: vscclrm            {s3, s4, s5, s6, s7, s8, vpr} @ encoding: [0xdf,0xec,0x06,0x1a]
// NOSEC: instruction requires: ARMv8-M Security Extensions
vscclrm {s3-s8, vpr}

// CHECK: vscclrm            {s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, vpr} @ encoding: [0x9f,0xec,0x0c,0x9a]
vscclrm {s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, vpr}

// CHECK: vscclrm            {s31, vpr} @ encoding: [0xdf,0xec,0x01,0xfa]
vscclrm {s31, vpr}

// CHECK: vscclrm            {d0, d1, vpr}  @ encoding: [0x9f,0xec,0x04,0x0b]
vscclrm {d0-d1, vpr}

// CHECK: vscclrm            {d0, d1, d2, d3, vpr}  @ encoding: [0x9f,0xec,0x08,0x0b]
vscclrm {d0-d3, vpr}

// CHECK: vscclrm            {d5, d6, d7, vpr}  @ encoding: [0x9f,0xec,0x06,0x5b]
vscclrm {d5-d7, vpr}

// CHECK: it                 hi @ encoding: [0x88,0xbf]
it hi
// CHECK: vscclrmhi          {s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, vpr} @ encoding: [0xdf,0xec,0x1d,0x1a]
vscclrmhi {s3-s31, vpr}

// CHECK: vscclrm            {vpr} @ encoding: [0x9f,0xec,0x00,0x0a]
vscclrm {vpr}

// CHECK: vscclrm            {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, vpr} @ encoding: [0x9f,0xec,0x40,0x0b]
vscclrm {d0-d31, vpr}

// CHECK: vscclrm            {d31, vpr} @ encoding: [0xdf,0xec,0x02,0xfb]
vscclrm {d31, vpr}

// CHECK: vscclrm            {s31, d16, vpr} @ encoding: [0xdf,0xec,0x03,0xfa]
vscclrm {s31, d16, vpr}

// CHECK: vscclrm            {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, vpr} @ encoding: [0x9f,0xec,0x40,0x0a]
vscclrm {s0-s31, d16-d31, vpr}

// CHECK: vscclrm            {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, vpr} @ encoding: [0x9f,0xec,0x40,0x0a]
vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, vpr}

// ERROR: non-contiguous register range
vscclrm {s0, s3-s4, vpr}

// ERROR: non-contiguous register range
vscclrm {s31, d16, s30, vpr}

// ERROR: register expected
vscclrm {s32, vpr}

// ERROR: register expected
vscclrm {d32, vpr}

// ERROR: register expected
vscclrm {s31-s32, vpr}

// ERROR: register expected
vscclrm {d31-d32, vpr}

// ERROR: invalid operand for instruction
vscclrm {s0-s1}

// ERROR: register list not in ascending order
vscclrm {vpr, s0}

// ERROR: register list not in ascending order
vscclrm {vpr, s31}

// ERROR: register list not in ascending order
vscclrm {vpr, d0}

// ERROR: register list not in ascending order
vscclrm {vpr, d31}

// ERROR: invalid register in register list
vscclrm {s0, d0, vpr}

// ERROR: invalid register in register list
vscclrm {s0, d1, vpr}

// ERROR: invalid register in register list
vscclrm {d16, s31, vpr}
