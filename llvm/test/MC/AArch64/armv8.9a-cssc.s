// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu                -mattr=+v8.8a < %s 2>&1 | FileCheck --check-prefix=NO-CSSC-ERR %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu                -mattr=+v9.3a < %s 2>&1 | FileCheck --check-prefix=NO-CSSC-ERR %s

            abs     x0, x1
// CHECK:   abs     x0, x1       // encoding: [0x20,0x20,0xc0,0xda]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            abs     w0, w1
// CHECK:   abs     w0, w1       // encoding: [0x20,0x20,0xc0,0x5a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            cnt     x0, x1
// CHECK:   cnt     x0, x1       // encoding: [0x20,0x1c,0xc0,0xda]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            cnt     w0, w1
// CHECK:   cnt     w0, w1       // encoding: [0x20,0x1c,0xc0,0x5a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            ctz     x0, x1
// CHECK:   ctz     x0, x1       // encoding: [0x20,0x18,0xc0,0xda]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            ctz     w0, w1
// CHECK:   ctz     w0, w1       // encoding: [0x20,0x18,0xc0,0x5a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc

            smax    x1, x2, x3
// CHECK:   smax    x1, x2, x3   // encoding: [0x41,0x60,0xc3,0x9a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smax    x1, x2, #3
// CHECK:   smax    x1, x2, #3   // encoding: [0x41,0x0c,0xc0,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smax    w1, w2, w3
// CHECK:   smax    w1, w2, w3   // encoding: [0x41,0x60,0xc3,0x1a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smax    w1, w2, #3
// CHECK:   smax    w1, w2, #3   // encoding: [0x41,0x0c,0xc0,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smin    x1, x2, x3
// CHECK:   smin    x1, x2, x3   // encoding: [0x41,0x68,0xc3,0x9a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smin    x1, x2, #3
// CHECK:   smin    x1, x2, #3   // encoding: [0x41,0x0c,0xc8,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smin    w1, w2, w3
// CHECK:   smin    w1, w2, w3   // encoding: [0x41,0x68,0xc3,0x1a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smin    w1, w2, #3
// CHECK:   smin    w1, w2, #3   // encoding: [0x41,0x0c,0xc8,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umax    x1, x2, x3
// CHECK:   umax    x1, x2, x3   // encoding: [0x41,0x64,0xc3,0x9a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umax    x1, x2, #3
// CHECK:   umax    x1, x2, #3   // encoding: [0x41,0x0c,0xc4,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umax    w1, w2, w3
// CHECK:   umax    w1, w2, w3   // encoding: [0x41,0x64,0xc3,0x1a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umax    w1, w2, #3
// CHECK:   umax    w1, w2, #3   // encoding: [0x41,0x0c,0xc4,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umin    x1, x2, x3
// CHECK:   umin    x1, x2, x3   // encoding: [0x41,0x6c,0xc3,0x9a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umin    x1, x2, #3
// CHECK:   umin    x1, x2, #3   // encoding: [0x41,0x0c,0xcc,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umin    w1, w2, w3
// CHECK:   umin    w1, w2, w3   // encoding: [0x41,0x6c,0xc3,0x1a]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umin    w1, w2, #3
// CHECK:   umin    w1, w2, #3   // encoding: [0x41,0x0c,0xcc,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc


            umax    wzr, wzr, #255
// CHECK:   umax    wzr, wzr, #255    // encoding: [0xff,0xff,0xc7,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umax    xzr, xzr, #255
// CHECK:   umax    xzr, xzr, #255    // encoding: [0xff,0xff,0xc7,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umin    xzr, xzr, #255
// CHECK:   umin    xzr, xzr, #255     // encoding: [0xff,0xff,0xcf,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            umin    wzr, wzr, #255
// CHECK:   umin    wzr, wzr, #255    // encoding: [0xff,0xff,0xcf,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smax    xzr, xzr, #-1
// CHECK:   smax    xzr, xzr, #-1     // encoding: [0xff,0xff,0xc3,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smax    wzr, wzr, #-1
// CHECK:   smax    wzr, wzr, #-1     // encoding: [0xff,0xff,0xc3,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smin    xzr, xzr, #-1
// CHECK:   smin    xzr, xzr, #-1     // encoding: [0xff,0xff,0xcb,0x91]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
            smin    wzr, wzr, #-1
// CHECK:   smin    wzr, wzr, #-1   // encoding: [0xff,0xff,0xcb,0x11]
// NO-CSSC-ERR: [[@LINE-2]]:13: error: instruction requires: cssc
