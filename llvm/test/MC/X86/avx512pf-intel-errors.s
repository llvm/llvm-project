// RUN: not llvm-mc -triple x86_64 -x86-asm-syntax=intel %s -o /dev/null 2>&1 | FileCheck %s

// Masks remain mandatory and nonzero; zeroing is not supported.
// The VSIB register width and explicit memory size must still match.

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: vgatherpf0dps [rax+zmm0]
vgatherpf0dps [rax+zmm0]

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: Register k0 can't be used as write mask
// CHECK-NEXT: vgatherpf0dps [rax+zmm0] {k0}
vgatherpf0dps [rax+zmm0] {k0}

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: vgatherpf0dps [rax+zmm0] {k1} {z}
vgatherpf0dps [rax+zmm0] {k1} {z}

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: vgatherpf0dps [rax+ymm0] {k1}
vgatherpf0dps [rax+ymm0] {k1}

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: vgatherpf0dpd [rax+zmm0] {k1}
vgatherpf0dpd [rax+zmm0] {k1}

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: vgatherpf0dps qword ptr [rax+zmm0] {k1}
vgatherpf0dps qword ptr [rax+zmm0] {k1}
