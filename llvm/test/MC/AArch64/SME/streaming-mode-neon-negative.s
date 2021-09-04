// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve,+bf16 2>&1 < %s| FileCheck %s

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
