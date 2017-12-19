// RUN: llvm-mc -triple=aarch64-none-linux-gnu -show-encoding < %s 2>&1 | FileCheck %s

bar:
        fred .req x5
// CHECK-NOT: ignoring redefinition of register alias 'fred'
        fred .req x5
        mov fred, x11
        .unreq fred
        fred .req w6
        mov w1, fred

        bob .req fred
        ada .req w1
        mov ada, bob
        .unreq bob
        .unreq fred
        .unreq ada
// CHECK: mov      x5, x11                // encoding: [0xe5,0x03,0x0b,0xaa]
// CHECK: mov      w1, w6                 // encoding: [0xe1,0x03,0x06,0x2a]
// CHECK: mov      w1, w6                 // encoding: [0xe1,0x03,0x06,0x2a]

        bob     .req b6
        hanah   .req h5
        sam     .req s4
        dora    .req d3
        quentin .req q2
        vesna   .req v1
        addv bob,     v0.8b
        mov  hanah,   v4.h[3]
        fadd s0,      sam,     sam
        fmov d2,      dora
        ldr  quentin, [sp]
        mov  v0.8b,   vesna.8b
// CHECK: addv    b6, v0.8b               // encoding: [0x06,0xb8,0x31,0x0e]
// CHECK: mov     h5, v4.h[3]             // encoding: [0x85,0x04,0x0e,0x5e]
// CHECK: fadd    s0, s4, s4              // encoding: [0x80,0x28,0x24,0x1e]
// CHECK: fmov    d2, d3                  // encoding: [0x62,0x40,0x60,0x1e]
// CHECK: ldr      q2, [sp]               // encoding: [0xe2,0x03,0xc0,0x3d]
// CHECK: mov             v0.8b, v1.8b    // encoding: [0x20,0x1c,0xa1,0x0e]

	peter .req x6
	add peter, x0, x0
	.unreq peter
// CHECK: add x6, x0, x0
