// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+sve < %s 2>&1 | FileCheck %s

//------------------------------------------------------------------------------
// Condition code diagnostics for SVE
//------------------------------------------------------------------------------

        b.nfirst lbl
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid condition code, did you mean nfrst?
// CHECK-NEXT:  b.nfirst lbl
