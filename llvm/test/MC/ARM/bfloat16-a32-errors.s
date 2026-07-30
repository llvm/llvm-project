// RUN: not llvm-mc -triple arm -mattr=+bf16,-neon %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NONEON,ALL
// RUN: not llvm-mc -triple arm -mattr=-bf16 %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NOBF16,ALL
// RUN: not llvm-mc -triple arm %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=NONEON,ALL
//
vdot.bf16 d3, d4, d5
vdot.bf16 q0, q1, q2
vdot.bf16 d3, d4, d5[1]
vdot.bf16 q0, q1, d5[1]
vmmla.bf16 q0, q1, q2
vcvt.bf16.f32 d1, q3
vcvtbeq.bf16.f32 s1, s3
vcvttne.bf16.f32 s1, s3
// NOBF16: error: instruction requires: BFloat16 floating point extension
// NOBF16-NEXT: vdot.bf16 d3, d4, d5
// NOBF16-NEXT: ^
// NOBF16-NEXT: error: instruction requires: BFloat16 floating point extension
// NOBF16-NEXT: vdot.bf16 q0, q1, q2
// NOBF16-NEXT: ^
// NOBF16-NEXT: error: instruction requires: BFloat16 floating point extension
// NOBF16-NEXT: vdot.bf16 d3, d4, d5[1]
// NOBF16-NEXT: ^
// NOBF16-NEXT: error: instruction requires: BFloat16 floating point extension
// NOBF16-NEXT: vdot.bf16 q0, q1, d5[1]
// NOBF16-NEXT: ^
// NOBF16-NEXT: error: instruction requires: BFloat16 floating point extension
// NOBF16-NEXT: vmmla.bf16 q0, q1, q2
// NOBF16-NEXT: ^
// NOBF16-NEXT: error: instruction requires: BFloat16 floating point extension
// NOBF16-NEXT: vcvt.bf16.f32 d1, q3
// NOBF16-NEXT: ^

// NONEON: error: instruction requires: BFloat16 floating point extension NEON
// NONEON-NEXT: vdot.bf16 d3, d4, d5
// NONEON-NEXT: ^
// NONEON-NEXT: error: instruction requires: BFloat16 floating point extension NEON
// NONEON-NEXT: vdot.bf16 q0, q1, q2
// NONEON-NEXT: ^
// NONEON-NEXT: error: instruction requires: BFloat16 floating point extension NEON
// NONEON-NEXT: vdot.bf16 d3, d4, d5[1]
// NONEON-NEXT: ^
// NONEON-NEXT: error: instruction requires: BFloat16 floating point extension NEON
// NONEON-NEXT: vdot.bf16 q0, q1, d5[1]
// NONEON-NEXT: ^
// NONEON-NEXT: error: instruction requires: BFloat16 floating point extension NEON
// NONEON-NEXT: vmmla.bf16 q0, q1, q2
// NONEON-NEXT: ^
// NONEON-NEXT: error: instruction requires: BFloat16 floating point extension NEON
// NONEON-NEXT: vcvt.bf16.f32 d1, q3
// NONEON-NEXT: ^


// ALL-NEXT: error: instruction requires: BFloat16 floating point extension
// ALL-NEXT: vcvtbeq.bf16.f32 s1, s3
// ALL-NEXT: ^
// ALL-NEXT: error: instruction requires: BFloat16 floating point extension
// ALL-NEXT: vcvttne.bf16.f32 s1, s3
// ALL-NEXT: ^
