// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f8f16,+sme-f8f32  2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector select register

fmlal    za.h[w8, 0:1, vgx2], {z0.h-z1.h}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlal    za.h[w8, 0:1, vgx2], {z0.h-z1.h}, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w11, 4:7], {z31.b-z2.b}, z15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: fmlal    za.h[w11, 4:7], {z31.b-z2.b}, z15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w11, 6:7, vgx2], {z28.b-z31.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlal    za.h[w11, 6:7, vgx2], {z28.b-z31.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.s[w11, 0:3], {z29.b-z30.b}, {z30.b-z31.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlall    za.s[w11, 0:3], {z29.b-z30.b}, {z30.b-z31.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.s[w11, 4:7], {z30.b-z0.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlall    za.s[w11, 4:7], {z30.b-z0.b}, z15.
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector select offset

fmlal   za.h[w11, 1:2], {z30.b-z31.b}, z15.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 6] or [0, 14] depending on the instruction, and the second immediate is immf + 1.
// CHECK-NEXT: fmlal   za.h[w11, 1:2], {z30.b-z31.b}, z15.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w11, 3:4], {z28.b-z31.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 6] or [0, 14] depending on the instruction, and the second immediate is immf + 1.
// CHECK-NEXT: fmlal    za.h[w11, 3:4], {z28.b-z31.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w11, 7:8, vgx4], {z28.b-z31.b}, {z4.b-z7.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 2 in the range [0, 6] or [0, 14] depending on the instruction, and the second immediate is immf + 1.
// CHECK-NEXT: fmlal    za.h[w11, 7:8, vgx4], {z28.b-z31.b}, {z4.b-z7.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w11, 3:6, vgx4], {z30.b-z31.b}, z15.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: fmlall  za.s[w11, 3:6, vgx4], {z30.b-z31.b}, z15.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w8, 3:6, vgx4], {z0.b-z3.b}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: fmlall  za.s[w8, 3:6, vgx4], {z0.b-z3.b}, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w11, 7:10, vgx4], {z30.b-z31.b}, {z12.b-z13.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector select offset must be an immediate range of the form <immf>:<imml>, where the first immediate is a multiple of 4 in the range [0, 4] or [0, 12] depending on the instruction, and the second immediate is immf + 3.
// CHECK-NEXT: fmlall  za.s[w11, 7:10, vgx4], {z30.b-z31.b}, {z12.b-z13.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

fmlal    za.h[w11, 4:7, vgx4], {z29.b-z1.b}, {z29.b-z1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fmlal    za.h[w11, 4:7, vgx4], {z29.b-z1.b}, {z29.b-z1.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w11, 4:7], {z30.b-z2.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fmlal    za.h[w11, 4:7], {z30.b-z2.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.s[w8, 0:1], {z31.b-z3.b}, {z31.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fmlall    za.s[w8, 0:1], {z31.b-z3.b}, {z31.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.s[w11, 6:7, vgx2], {z30.b-z31.b}, {z0.b-z4.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: fmlall    za.s[w11, 6:7, vgx2], {z30.b-z31.b}, {z0.b-z4.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix
fmlal    za.d[w11, 4:5, vgx4], {z31.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fmlal    za.d[w11, 4:5, vgx4], {z31.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za[w11, 2:3], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fmlal    za[w11, 2:3], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.b[w11, 6:7], {z31.b-z0.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fmlal    za.b[w11, 6:7], {z31.b-z0.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.b[w11, 6:7, vgx2], {z30.h-z31.h}, {z30.h-z31.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fmlall    za.b[w11, 6:7, vgx2], {z30.h-z31.h}, {z30.h-z31.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za[w11, 4:7, vgx4], {z31.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fmlall    za[w11, 4:7, vgx4], {z31.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.d[w11, 12:15], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: fmlall    za.d[w11, 12:15], {z28.b-z31.b}, {z28.b-z31.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

fmlal    za.h[w7, 4:7, vgx4], {z31.b-z2.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fmlal    za.h[w7, 4:7, vgx4], {z31.b-z2.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w, 0:1, vgx2], {z0.b-z1.b}, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fmlal    za.h[w, 0:1, vgx2], {z0.b-z1.b}, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.s[w12, 0:3], {z0.b-z3.b}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: fmlall    za.s[w12, 0:3], {z0.b-z3.b}, {z0.b-z3.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector or single-vector register

fmlal za.h[w8, 0:1], {z0.b-z1.b}, z16.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: fmlal za.h[w8, 0:1], {z0.b-z1.b}, z16.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal   za.h[w9, 14:15], z31.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: fmlal   za.h[w9, 14:15], z31.b, z16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w11, 8:11], z9.b, z16.b[13]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: fmlall  za.s[w11, 8:11], z9.b, z16.b[13]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w11, 12:15], z31.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.b..z15.b
// CHECK-NEXT: fmlall  za.s[w11, 12:15], z31.b, z16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector grouping

fmlal    za.h[w11, 10:11], {z28.b-z31.b}, {z0.b-z2.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlal    za.h[w11, 10:11], {z28.b-z31.b}, {z0.b-z2.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall    za.s[w11, 4:7, vgx4], {z31.b-z0.b}, z15.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlall    za.s[w11, 4:7, vgx4], {z31.b-z0.b}, z15.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

fmlal   za.h[w11, 14:15], z31.b, z15.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlal   za.h[w11, 14:15], z31.b, z15.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlal    za.h[w11, 2:3], {z30.b-z31.b}, z15.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlal    za.h[w11, 2:3], {z30.b-z31.b}, z15.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w9, 12:15], z12.b, z11.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlall  za.s[w9, 12:15], z12.b, z11.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlall  za.s[w8, 4:7], {z16.b-z19.b}, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlall  za.s[w8, 4:7], {z16.b-z19.b}, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
