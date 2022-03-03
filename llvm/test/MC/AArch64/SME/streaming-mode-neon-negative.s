// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=-neon,+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Check FABD is illegal in streaming mode

fabd s0, s1, s2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: fabd s0, s1, s2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Check non-scalar v8.6a BFloat16 instructions are illegal in streaming mode

bfcvtn v5.4h, v5.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: bfcvtn v5.4h, v5.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Check non-zero index is illegal in streaming mode
// ------------------------------------------------------------------------- //
// SMOV 8-bit to 32-bit

smov w0, v0.b[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov w0, v0.b[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov w0, v0.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov w0, v0.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov w0, v0.b[15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov w0, v0.b[15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// SMOV 8-bit to 64-bit

smov x0, v0.b[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.b[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov x0, v0.b[6]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.b[6]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov x0, v0.b[12]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.b[12]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// SMOV 16-bit to 32-bit

smov w0, v0.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov w0, v0.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov w0, v0.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov w0, v0.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov w0, v0.h[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov w0, v0.h[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// SMOV 16-bit to 64-bit

smov x0, v0.h[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.h[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov x0, v0.h[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.h[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov x0, v0.h[6]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.h[6]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// SMOV 32-bit to 64-bit

smov x0, v0.s[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.s[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov x0, v0.s[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.s[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

smov x0, v0.s[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: smov x0, v0.s[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// UMOV 8-bit to 32-bit

umov w0, v0.b[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.b[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umov w0, v0.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umov w0, v0.b[15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.b[15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// UMOV 16-bit to 32-bit

umov w0, v0.h[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.h[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umov w0, v0.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umov w0, v0.h[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.h[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// UMOV 32-bit to 32-bit

umov w0, v0.s[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.s[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umov w0, v0.s[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.s[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umov w0, v0.s[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov w0, v0.s[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// UMOV 64-bit to 64-bit

umov x0, v0.d[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: neon
// CHECK-NEXT: umov x0, v0.d[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
