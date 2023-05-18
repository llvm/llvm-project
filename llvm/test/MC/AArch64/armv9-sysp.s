// +tbl-rmi required for RIPA*/RVA*
// +xs required for *NXS

// RUN: not llvm-mc -triple aarch64 -mattr=+d128,+tlb-rmi,+xs -show-encoding %s -o - 2> %t | FileCheck %s
// RUN: FileCheck %s --input-file=%t --check-prefix=ERRORS

// RUN: not llvm-mc -triple aarch64 -mattr=+tlb-rmi,+xs -show-encoding %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR-NO-D128

// sysp #<op1>, <Cn>, <Cm>, #<op2>{, <Xt1>, <Xt2>}
// registers with 128-bit formats (op0, op1, Cn, Cm, op2)
// For sysp, op0 is 0

          sysp #0, c2, c0, #0, x0, x1          // TTBR0_EL1     3  0  2  0  0
// CHECK: sysp #0, c2, c0, #0, x0, x1          // encoding: [0x00,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #1, x0, x1          // TTBR1_EL1     3  0  2  0  1
// CHECK: sysp #0, c2, c0, #1, x0, x1          // encoding: [0x20,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c7, c4, #0, x0, x1          // PAR_EL1       3  0  7  4  0
// CHECK: sysp #0, c7, c4, #0, x0, x1          // encoding: [0x00,0x74,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c13, c0, #3, x0, x1         // RCWSMASK_EL1  3  0 13  0  3
// CHECK: sysp #0, c13, c0, #3, x0, x1         // encoding: [0x60,0xd0,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c13, c0, #6, x0, x1         // RCWMASK_EL1   3  0 13  0  6
// CHECK: sysp #0, c13, c0, #6, x0, x1         // encoding: [0xc0,0xd0,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #4, c2, c0, #0, x0, x1          // TTBR0_EL2     3  4  2  0  0
// CHECK: sysp #4, c2, c0, #0, x0, x1          // encoding: [0x00,0x20,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #4, c2, c0, #1, x0, x1          // TTBR1_EL2     3  4  2  0  1
// CHECK: sysp #4, c2, c0, #1, x0, x1          // encoding: [0x20,0x20,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #4, c2, c1, #0, x0, x1          // VTTBR_EL2     3  4  2  1  0
// CHECK: sysp #4, c2, c1, #0, x0, x1          // encoding: [0x00,0x21,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128


          sysp #0, c2, c0, #0, x0, x1
// CHECK: sysp #0, c2, c0, #0, x0, x1          // encoding: [0x00,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #1, x0, x1
// CHECK: sysp #0, c2, c0, #1, x0, x1          // encoding: [0x20,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c7, c4, #0, x0, x1
// CHECK: sysp #0, c7, c4, #0, x0, x1          // encoding: [0x00,0x74,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c13, c0, #3, x0, x1
// CHECK: sysp #0, c13, c0, #3, x0, x1         // encoding: [0x60,0xd0,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c13, c0, #6, x0, x1
// CHECK: sysp #0, c13, c0, #6, x0, x1         // encoding: [0xc0,0xd0,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #4, c2, c0, #0, x0, x1
// CHECK: sysp #4, c2, c0, #0, x0, x1          // encoding: [0x00,0x20,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #4, c2, c0, #1, x0, x1
// CHECK: sysp #4, c2, c0, #1, x0, x1          // encoding: [0x20,0x20,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #4, c2, c1, #0, x0, x1
// CHECK: sysp #4, c2, c1, #0, x0, x1          // encoding: [0x00,0x21,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          sysp #0, c2, c0, #0, x0, x1
// CHECK: sysp #0, c2, c0, #0, x0, x1          // encoding: [0x00,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x2, x3
// CHECK: sysp #0, c2, c0, #0, x2, x3          // encoding: [0x02,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x4, x5
// CHECK: sysp #0, c2, c0, #0, x4, x5          // encoding: [0x04,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x6, x7
// CHECK: sysp #0, c2, c0, #0, x6, x7          // encoding: [0x06,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x8, x9
// CHECK: sysp #0, c2, c0, #0, x8, x9          // encoding: [0x08,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x10, x11
// CHECK: sysp #0, c2, c0, #0, x10, x11        // encoding: [0x0a,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x12, x13
// CHECK: sysp #0, c2, c0, #0, x12, x13        // encoding: [0x0c,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x14, x15
// CHECK: sysp #0, c2, c0, #0, x14, x15        // encoding: [0x0e,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x16, x17
// CHECK: sysp #0, c2, c0, #0, x16, x17        // encoding: [0x10,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x18, x19
// CHECK: sysp #0, c2, c0, #0, x18, x19        // encoding: [0x12,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x20, x21
// CHECK: sysp #0, c2, c0, #0, x20, x21        // encoding: [0x14,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x22, x23
// CHECK: sysp #0, c2, c0, #0, x22, x23        // encoding: [0x16,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x24, x25
// CHECK: sysp #0, c2, c0, #0, x24, x25        // encoding: [0x18,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x26, x27
// CHECK: sysp #0, c2, c0, #0, x26, x27        // encoding: [0x1a,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x28, x29
// CHECK: sysp #0, c2, c0, #0, x28, x29        // encoding: [0x1c,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x30, x31
// CHECK: sysp #0, c2, c0, #0, x30, xzr        // encoding: [0x1e,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          sysp #0, c2, c0, #0, x31, x31
// CHECK: sysp #0, c2, c0, #0                  // encoding: [0x1f,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, xzr, xzr
// CHECK: sysp #0, c2, c0, #0                  // encoding: [0x1f,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, x31, xzr
// CHECK: sysp #0, c2, c0, #0                  // encoding: [0x1f,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0, xzr, x31
// CHECK: sysp #0, c2, c0, #0                  // encoding: [0x1f,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          sysp #0, c2, c0, #0
// CHECK: sysp #0, c2, c0, #0                  // encoding: [0x1f,0x20,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128


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


          tlbip IPAS2E1, x4, x5
// CHECK: tlbip ipas2e1, x4, x5                 // encoding: [0x24,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2E1NXS, x4, x5
// CHECK: tlbip ipas2e1nxs, x4, x5              // encoding: [0x24,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2E1IS, x4, x5
// CHECK: tlbip ipas2e1is, x4, x5               // encoding: [0x24,0x80,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2E1ISNXS, x4, x5
// CHECK: tlbip ipas2e1isnxs, x4, x5            // encoding: [0x24,0x90,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2E1OS, x4, x5
// CHECK: tlbip ipas2e1os, x4, x5               // encoding: [0x04,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2E1OSNXS, x4, x5
// CHECK: tlbip ipas2e1osnxs, x4, x5            // encoding: [0x04,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2LE1, x4, x5
// CHECK: tlbip ipas2le1, x4, x5                // encoding: [0xa4,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2LE1NXS, x4, x5
// CHECK: tlbip ipas2le1nxs, x4, x5             // encoding: [0xa4,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2LE1IS, x4, x5
// CHECK: tlbip ipas2le1is, x4, x5              // encoding: [0xa4,0x80,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2LE1ISNXS, x4, x5
// CHECK: tlbip ipas2le1isnxs, x4, x5           // encoding: [0xa4,0x90,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2LE1OS, x4, x5
// CHECK: tlbip ipas2le1os, x4, x5              // encoding: [0x84,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip IPAS2LE1OSNXS, x4, x5
// CHECK: tlbip ipas2le1osnxs, x4, x5           // encoding: [0x84,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128


          tlbip VAE1, x8, x9
// CHECK: tlbip vae1, x8, x9                    // encoding: [0x28,0x87,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE1NXS, x8, x9
// CHECK: tlbip vae1nxs, x8, x9                 // encoding: [0x28,0x97,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE1IS, x8, x9
// CHECK: tlbip vae1is, x8, x9                  // encoding: [0x28,0x83,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE1ISNXS, x8, x9
// CHECK: tlbip vae1isnxs, x8, x9               // encoding: [0x28,0x93,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE1OS, x8, x9
// CHECK: tlbip vae1os, x8, x9                  // encoding: [0x28,0x81,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE1OSNXS, x8, x9
// CHECK: tlbip vae1osnxs, x8, x9               // encoding: [0x28,0x91,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE1, x8, x9
// CHECK: tlbip vale1, x8, x9                   // encoding: [0xa8,0x87,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE1NXS, x8, x9
// CHECK: tlbip vale1nxs, x8, x9                // encoding: [0xa8,0x97,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE1IS, x8, x9
// CHECK: tlbip vale1is, x8, x9                 // encoding: [0xa8,0x83,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE1ISNXS, x8, x9
// CHECK: tlbip vale1isnxs, x8, x9              // encoding: [0xa8,0x93,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE1OS, x8, x9
// CHECK: tlbip vale1os, x8, x9                 // encoding: [0xa8,0x81,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE1OSNXS, x8, x9
// CHECK: tlbip vale1osnxs, x8, x9              // encoding: [0xa8,0x91,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAAE1, x8, x9
// CHECK: tlbip vaae1, x8, x9                   // encoding: [0x68,0x87,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAAE1NXS, x8, x9
// CHECK: tlbip vaae1nxs, x8, x9                // encoding: [0x68,0x97,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAAE1IS, x8, x9
// CHECK: tlbip vaae1is, x8, x9                 // encoding: [0x68,0x83,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAAE1ISNXS, x8, x9
// CHECK: tlbip vaae1isnxs, x8, x9              // encoding: [0x68,0x93,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAAE1OS, x8, x9
// CHECK: tlbip vaae1os, x8, x9                 // encoding: [0x68,0x81,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAAE1OSNXS, x8, x9
// CHECK: tlbip vaae1osnxs, x8, x9              // encoding: [0x68,0x91,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAALE1, x8, x9
// CHECK: tlbip vaale1, x8, x9                  // encoding: [0xe8,0x87,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAALE1NXS, x8, x9
// CHECK: tlbip vaale1nxs, x8, x9               // encoding: [0xe8,0x97,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAALE1IS, x8, x9
// CHECK: tlbip vaale1is, x8, x9                // encoding: [0xe8,0x83,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAALE1ISNXS, x8, x9
// CHECK: tlbip vaale1isnxs, x8, x9             // encoding: [0xe8,0x93,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAALE1OS, x8, x9
// CHECK: tlbip vaale1os, x8, x9                // encoding: [0xe8,0x81,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAALE1OSNXS, x8, x9
// CHECK: tlbip vaale1osnxs, x8, x9             // encoding: [0xe8,0x91,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          tlbip VAE2, x14, x15
// CHECK: tlbip vae2, x14, x15                    // encoding: [0x2e,0x87,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE2NXS, x14, x15
// CHECK: tlbip vae2nxs, x14, x15                 // encoding: [0x2e,0x97,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE2IS, x14, x15
// CHECK: tlbip vae2is, x14, x15                  // encoding: [0x2e,0x83,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE2ISNXS, x14, x15
// CHECK: tlbip vae2isnxs, x14, x15               // encoding: [0x2e,0x93,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE2OS, x14, x15
// CHECK: tlbip vae2os, x14, x15                  // encoding: [0x2e,0x81,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE2OSNXS, x14, x15
// CHECK: tlbip vae2osnxs, x14, x15               // encoding: [0x2e,0x91,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE2, x14, x15
// CHECK: tlbip vale2, x14, x15                   // encoding: [0xae,0x87,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE2NXS, x14, x15
// CHECK: tlbip vale2nxs, x14, x15                // encoding: [0xae,0x97,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE2IS, x14, x15
// CHECK: tlbip vale2is, x14, x15                 // encoding: [0xae,0x83,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE2ISNXS, x14, x15
// CHECK: tlbip vale2isnxs, x14, x15              // encoding: [0xae,0x93,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE2OS, x14, x15
// CHECK: tlbip vale2os, x14, x15                 // encoding: [0xae,0x81,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE2OSNXS, x14, x15
// CHECK: tlbip vale2osnxs, x14, x15              // encoding: [0xae,0x91,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          tlbip VAE3, x24, x25
// CHECK: tlbip vae3, x24, x25                    // encoding: [0x38,0x87,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE3NXS, x24, x25
// CHECK: tlbip vae3nxs, x24, x25                 // encoding: [0x38,0x97,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE3IS, x24, x25
// CHECK: tlbip vae3is, x24, x25                  // encoding: [0x38,0x83,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE3ISNXS, x24, x25
// CHECK: tlbip vae3isnxs, x24, x25               // encoding: [0x38,0x93,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE3OS, x24, x25
// CHECK: tlbip vae3os, x24, x25                  // encoding: [0x38,0x81,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VAE3OSNXS, x24, x25
// CHECK: tlbip vae3osnxs, x24, x25               // encoding: [0x38,0x91,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE3, x24, x25
// CHECK: tlbip vale3, x24, x25                   // encoding: [0xb8,0x87,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE3NXS, x24, x25
// CHECK: tlbip vale3nxs, x24, x25                // encoding: [0xb8,0x97,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE3IS, x24, x25
// CHECK: tlbip vale3is, x24, x25                 // encoding: [0xb8,0x83,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE3ISNXS, x24, x25
// CHECK: tlbip vale3isnxs, x24, x25              // encoding: [0xb8,0x93,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE3OS, x24, x25
// CHECK: tlbip vale3os, x24, x25                 // encoding: [0xb8,0x81,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip VALE3OSNXS, x24, x25
// CHECK: tlbip vale3osnxs, x24, x25              // encoding: [0xb8,0x91,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128


          tlbip RVAE1, x18, x19
// CHECK: tlbip rvae1, x18, x19                   // encoding: [0x32,0x86,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE1NXS, x18, x19
// CHECK: tlbip rvae1nxs, x18, x19                // encoding: [0x32,0x96,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE1IS, x18, x19
// CHECK: tlbip rvae1is, x18, x19                 // encoding: [0x32,0x82,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE1ISNXS, x18, x19
// CHECK: tlbip rvae1isnxs, x18, x19              // encoding: [0x32,0x92,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE1OS, x18, x19
// CHECK: tlbip rvae1os, x18, x19                 // encoding: [0x32,0x85,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE1OSNXS, x18, x19
// CHECK: tlbip rvae1osnxs, x18, x19              // encoding: [0x32,0x95,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAAE1, x18, x19
// CHECK: tlbip rvaae1, x18, x19                  // encoding: [0x72,0x86,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAAE1NXS, x18, x19
// CHECK: tlbip rvaae1nxs, x18, x19               // encoding: [0x72,0x96,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAAE1IS, x18, x19
// CHECK: tlbip rvaae1is, x18, x19                // encoding: [0x72,0x82,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAAE1ISNXS, x18, x19
// CHECK: tlbip rvaae1isnxs, x18, x19             // encoding: [0x72,0x92,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAAE1OS, x18, x19
// CHECK: tlbip rvaae1os, x18, x19                // encoding: [0x72,0x85,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAAE1OSNXS, x18, x19
// CHECK: tlbip rvaae1osnxs, x18, x19             // encoding: [0x72,0x95,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE1, x18, x19
// CHECK: tlbip rvale1, x18, x19                  // encoding: [0xb2,0x86,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE1NXS, x18, x19
// CHECK: tlbip rvale1nxs, x18, x19               // encoding: [0xb2,0x96,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE1IS, x18, x19
// CHECK: tlbip rvale1is, x18, x19                // encoding: [0xb2,0x82,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE1ISNXS, x18, x19
// CHECK: tlbip rvale1isnxs, x18, x19             // encoding: [0xb2,0x92,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE1OS, x18, x19
// CHECK: tlbip rvale1os, x18, x19                // encoding: [0xb2,0x85,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE1OSNXS, x18, x19
// CHECK: tlbip rvale1osnxs, x18, x19             // encoding: [0xb2,0x95,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAALE1, x18, x19
// CHECK: tlbip rvaale1, x18, x19                 // encoding: [0xf2,0x86,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAALE1NXS, x18, x19
// CHECK: tlbip rvaale1nxs, x18, x19              // encoding: [0xf2,0x96,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAALE1IS, x18, x19
// CHECK: tlbip rvaale1is, x18, x19               // encoding: [0xf2,0x82,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAALE1ISNXS, x18, x19
// CHECK: tlbip rvaale1isnxs, x18, x19            // encoding: [0xf2,0x92,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAALE1OS, x18, x19
// CHECK: tlbip rvaale1os, x18, x19               // encoding: [0xf2,0x85,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAALE1OSNXS, x18, x19
// CHECK: tlbip rvaale1osnxs, x18, x19            // encoding: [0xf2,0x95,0x48,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          tlbip RVAE2, x28, x29
// CHECK: tlbip rvae2, x28, x29                   // encoding: [0x3c,0x86,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE2NXS, x28, x29
// CHECK: tlbip rvae2nxs, x28, x29                // encoding: [0x3c,0x96,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE2IS, x28, x29
// CHECK: tlbip rvae2is, x28, x29                 // encoding: [0x3c,0x82,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE2ISNXS, x28, x29
// CHECK: tlbip rvae2isnxs, x28, x29              // encoding: [0x3c,0x92,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE2OS, x28, x29
// CHECK: tlbip rvae2os, x28, x29                 // encoding: [0x3c,0x85,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE2OSNXS, x28, x29
// CHECK: tlbip rvae2osnxs, x28, x29              // encoding: [0x3c,0x95,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE2, x28, x29
// CHECK: tlbip rvale2, x28, x29                  // encoding: [0xbc,0x86,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE2NXS, x28, x29
// CHECK: tlbip rvale2nxs, x28, x29               // encoding: [0xbc,0x96,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE2IS, x28, x29
// CHECK: tlbip rvale2is, x28, x29                // encoding: [0xbc,0x82,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE2ISNXS, x28, x29
// CHECK: tlbip rvale2isnxs, x28, x29             // encoding: [0xbc,0x92,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE2OS, x28, x29
// CHECK: tlbip rvale2os, x28, x29                // encoding: [0xbc,0x85,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE2OSNXS, x28, x29
// CHECK: tlbip rvale2osnxs, x28, x29             // encoding: [0xbc,0x95,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          tlbip RVAE3, x10, x11
// CHECK: tlbip rvae3, x10, x11                   // encoding: [0x2a,0x86,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE3NXS, x10, x11
// CHECK: tlbip rvae3nxs, x10, x11                // encoding: [0x2a,0x96,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE3IS, x10, x11
// CHECK: tlbip rvae3is, x10, x11                 // encoding: [0x2a,0x82,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE3ISNXS, x10, x11
// CHECK: tlbip rvae3isnxs, x10, x11              // encoding: [0x2a,0x92,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE3OS, x10, x11
// CHECK: tlbip rvae3os, x10, x11                 // encoding: [0x2a,0x85,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVAE3OSNXS, x10, x11
// CHECK: tlbip rvae3osnxs, x10, x11              // encoding: [0x2a,0x95,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE3, x10, x11
// CHECK: tlbip rvale3, x10, x11                  // encoding: [0xaa,0x86,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE3NXS, x10, x11
// CHECK: tlbip rvale3nxs, x10, x11               // encoding: [0xaa,0x96,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE3IS, x10, x11
// CHECK: tlbip rvale3is, x10, x11                // encoding: [0xaa,0x82,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE3ISNXS, x10, x11
// CHECK: tlbip rvale3isnxs, x10, x11             // encoding: [0xaa,0x92,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE3OS, x10, x11
// CHECK: tlbip rvale3os, x10, x11                // encoding: [0xaa,0x85,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RVALE3OSNXS, x10, x11
// CHECK: tlbip rvale3osnxs, x10, x11             // encoding: [0xaa,0x95,0x4e,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128


          tlbip RIPAS2E1, x20, x21
// CHECK: tlbip ripas2e1, x20, x21                // encoding: [0x54,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2E1NXS, x20, x21
// CHECK: tlbip ripas2e1nxs, x20, x21             // encoding: [0x54,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2E1IS, x20, x21
// CHECK: tlbip ripas2e1is, x20, x21              // encoding: [0x54,0x80,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2E1ISNXS, x20, x21
// CHECK: tlbip ripas2e1isnxs, x20, x21           // encoding: [0x54,0x90,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2E1OS, x20, x21
// CHECK: tlbip ripas2e1os, x20, x21              // encoding: [0x74,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2E1OSNXS, x20, x21
// CHECK: tlbip ripas2e1osnxs, x20, x21           // encoding: [0x74,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1, x20, x21
// CHECK: tlbip ripas2le1, x20, x21               // encoding: [0xd4,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1NXS, x20, x21
// CHECK: tlbip ripas2le1nxs, x20, x21            // encoding: [0xd4,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1IS, x20, x21
// CHECK: tlbip ripas2le1is, x20, x21             // encoding: [0xd4,0x80,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1ISNXS, x20, x21
// CHECK: tlbip ripas2le1isnxs, x20, x21          // encoding: [0xd4,0x90,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1OS, x20, x21
// CHECK: tlbip ripas2le1os, x20, x21             // encoding: [0xf4,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1OSNXS, x20, x21
// CHECK: tlbip ripas2le1osnxs, x20, x21          // encoding: [0xf4,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128

          tlbip RIPAS2LE1OS, xzr, xzr
// CHECK: tlbip ripas2le1os, xzr, xzr             // encoding: [0xff,0x84,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
          tlbip RIPAS2LE1OSNXS, xzr, xzr
// CHECK: tlbip ripas2le1osnxs, xzr, xzr          // encoding: [0xff,0x94,0x4c,0xd5]
// ERROR-NO-D128: [[@LINE-2]]:11: error: instruction requires: d128
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
