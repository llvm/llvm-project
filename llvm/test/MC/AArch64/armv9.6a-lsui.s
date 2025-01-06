// RUN: llvm-mc -triple aarch64 -mattr=+lsui -show-encoding %s  | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding %s 2>&1  | FileCheck %s --check-prefix=ERROR

_func:
// CHECK: _func:
//------------------------------------------------------------------------------
// Unprivileged load/store operations
//------------------------------------------------------------------------------
  ldtxr       x9, [sp]
// CHECK: ldtxr	x9, [sp]                        // encoding: [0xe9,0x7f,0x5f,0xc9]
// ERROR: error: instruction requires: lsui
  ldtxr       x9, [sp, #0]
// CHECK: ldtxr	x9, [sp]                        // encoding: [0xe9,0x7f,0x5f,0xc9]
// ERROR: error: instruction requires: lsui
  ldtxr       x10, [x11]
// CHECK: ldtxr	x10, [x11]                      // encoding: [0x6a,0x7d,0x5f,0xc9]
// ERROR: error: instruction requires: lsui
  ldtxr       x10, [x11, #0]
// CHECK: ldtxr	x10, [x11]                      // encoding: [0x6a,0x7d,0x5f,0xc9]
// ERROR: error: instruction requires: lsui

  ldatxr      x9, [sp]
// CHECK: ldatxr	x9, [sp]                        // encoding: [0xe9,0xff,0x5f,0xc9]
// ERROR: error: instruction requires: lsui
  ldatxr      x10, [x11]
// CHECK: ldatxr	x10, [x11]                      // encoding: [0x6a,0xfd,0x5f,0xc9]
// ERROR: error: instruction requires: lsui

  sttxr       wzr, w4, [sp]
// CHECK: sttxr	wzr, w4, [sp]                   // encoding: [0xe4,0x7f,0x1f,0x89]
// ERROR: error: instruction requires: lsui
  sttxr       wzr, w4, [sp, #0]
// CHECK: sttxr	wzr, w4, [sp]                   // encoding: [0xe4,0x7f,0x1f,0x89]
// ERROR: error: instruction requires: lsui
  sttxr       w5, x6, [x7]
// CHECK: sttxr	w5, x6, [x7]                    // encoding: [0xe6,0x7c,0x05,0xc9]
// ERROR: error: instruction requires: lsui
  sttxr       w5, x6, [x7, #0]
// CHECK: sttxr	w5, x6, [x7]                    // encoding: [0xe6,0x7c,0x05,0xc9]
// ERROR: error: instruction requires: lsui

  stltxr      w2, w4, [sp]
// CHECK: stltxr	w2, w4, [sp]                    // encoding: [0xe4,0xff,0x02,0x89]
// ERROR: error: instruction requires: lsui
  stltxr      w5, x6, [x7]
// CHECK: stltxr	w5, x6, [x7]                    // encoding: [0xe6,0xfc,0x05,0xc9]
// ERROR: error: instruction requires: lsui

//------------------------------------------------------------------------------
// Unprivileged load/store register pair (offset)
//------------------------------------------------------------------------------

  ldtp       x21, x29, [x2, #504]
// CHECK: ldtp	x21, x29, [x2, #504]            // encoding: [0x55,0xf4,0x5f,0xe9]
// ERROR: instruction requires: lsui
  ldtp       x22, x23, [x3, #-512]
// CHECK: ldtp	x22, x23, [x3, #-512]           // encoding: [0x76,0x5c,0x60,0xe9]
// ERROR: instruction requires: lsui
  ldtp       x24, x25, [x4, #8]
// CHECK: ldtp	x24, x25, [x4, #8]              // encoding: [0x98,0xe4,0x40,0xe9]
// ERROR: instruction requires: lsui

  sttp       x3, x5, [sp], #16
// CHECK: sttp	x3, x5, [sp], #16               // encoding: [0xe3,0x17,0x81,0xe8]
// ERROR: instruction requires: lsui
  sttp       x3, x5, [sp, #8]!
// CHECK: sttp	x3, x5, [sp, #8]!               // encoding: [0xe3,0x97,0x80,0xe9]
// ERROR: instruction requires: lsui

  sttp       q3, q5, [sp]
// CHECK: sttp	q3, q5, [sp]                    // encoding: [0xe3,0x17,0x00,0xed]
// ERROR: instruction requires: lsui
  sttp       q17, q19, [sp, #1008]
// CHECK: sttp	q17, q19, [sp, #1008]           // encoding: [0xf1,0xcf,0x1f,0xed]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// Load/store register pair (post-indexed)
//------------------------------------------------------------------------------

  ldtp       x21, x29, [x2], #504
// CHECK: ldtp	x21, x29, [x2], #504            // encoding: [0x55,0xf4,0xdf,0xe8]
// ERROR: instruction requires: lsui
  ldtp       x22, x23, [x3], #-512
// CHECK: ldtp	x22, x23, [x3], #-512           // encoding: [0x76,0x5c,0xe0,0xe8]
// ERROR: instruction requires: lsui
  ldtp       x24, x25, [x4], #8
// CHECK: ldtp	x24, x25, [x4], #8              // encoding: [0x98,0xe4,0xc0,0xe8]
// ERROR: instruction requires: lsui

  sttp       q3, q5, [sp], #0
// CHECK: sttp	q3, q5, [sp], #0                // encoding: [0xe3,0x17,0x80,0xec]
// ERROR: instruction requires: lsui
  sttp       q17, q19, [sp], #1008
// CHECK: sttp	q17, q19, [sp], #1008           // encoding: [0xf1,0xcf,0x9f,0xec]
// ERROR: instruction requires: lsui
  ldtp       q23, q29, [x1], #-1024
// CHECK: ldtp	q23, q29, [x1], #-1024          // encoding: [0x37,0x74,0xe0,0xec]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// Load/store register pair (pre-indexed)
//------------------------------------------------------------------------------
  ldtp       x21, x29, [x2, #504]!
// CHECK: ldtp	x21, x29, [x2, #504]!           // encoding: [0x55,0xf4,0xdf,0xe9]
// ERROR: instruction requires: lsui
  ldtp       x22, x23, [x3, #-512]!
// CHECK: ldtp	x22, x23, [x3, #-512]!          // encoding: [0x76,0x5c,0xe0,0xe9]
// ERROR: instruction requires: lsui
  ldtp       x24, x25, [x4, #8]!
// CHECK: ldtp	x24, x25, [x4, #8]!             // encoding: [0x98,0xe4,0xc0,0xe9]
// ERROR: instruction requires: lsui

  sttp       q3, q5, [sp, #0]!
// CHECK: sttp	q3, q5, [sp, #0]!               // encoding: [0xe3,0x17,0x80,0xed]
// ERROR: instruction requires: lsui
  sttp       q17, q19, [sp, #1008]!
// CHECK: sttp	q17, q19, [sp, #1008]!          // encoding: [0xf1,0xcf,0x9f,0xed]
// ERROR: instruction requires: lsui
  ldtp       q23, q29, [x1, #-1024]!
// CHECK: ldtp	q23, q29, [x1, #-1024]!         // encoding: [0x37,0x74,0xe0,0xed]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// CAS(P)T instructions
//------------------------------------------------------------------------------
  //64 bits
  cast       x0, x1, [x2]
// CHECK: cast	x0, x1, [x2]                    // encoding: [0x41,0x7c,0x80,0xc9]
// ERROR: instruction requires: lsui
  cast       x0, x1, [sp, #0]
// CHECK: cast	x0, x1, [sp]                    // encoding: [0xe1,0x7f,0x80,0xc9]
// ERROR: instruction requires: lsui
  casat      x0, x1, [x2]
// CHECK: casat	x0, x1, [x2]                    // encoding: [0x41,0x7c,0xc0,0xc9]
// ERROR: instruction requires: lsui
  casat      x0, x1, [sp, #0]
// CHECK: casat	x0, x1, [sp]                    // encoding: [0xe1,0x7f,0xc0,0xc9]
// ERROR: instruction requires: lsui
  casalt     x0, x1, [x2]
// CHECK: casalt	x0, x1, [x2]                    // encoding: [0x41,0xfc,0xc0,0xc9]
// ERROR: instruction requires: lsui
  casalt     x0, x1, [sp, #0]
// CHECK: casalt	x0, x1, [sp]                    // encoding: [0xe1,0xff,0xc0,0xc9]
// ERROR: instruction requires: lsui
  caslt      x0, x1, [x2]
// CHECK: caslt	x0, x1, [x2]                    // encoding: [0x41,0xfc,0x80,0xc9]
// ERROR: instruction requires: lsui
  caslt      x0, x1, [sp, #0]
// CHECK: caslt	x0, x1, [sp]                    // encoding: [0xe1,0xff,0x80,0xc9]
// ERROR: instruction requires: lsui

  //CASP instruction
  caspt      x0, x1, x2, x3, [x4]
// CHECK: caspt	x0, x1, x2, x3, [x4]            // encoding: [0x82,0x7c,0x80,0x49]
// ERROR: instruction requires: lsui
  caspt      x0, x1, x2, x3, [sp, #0]
// CHECK: caspt	x0, x1, x2, x3, [sp]            // encoding: [0xe2,0x7f,0x80,0x49]
// ERROR: instruction requires: lsui
  caspat     x0, x1, x2, x3, [x4]
// CHECK: caspat	x0, x1, x2, x3, [x4]            // encoding: [0x82,0x7c,0xc0,0x49]
// ERROR: instruction requires: lsui
  caspat     x0, x1, x2, x3, [sp, #0]
// CHECK: caspat	x0, x1, x2, x3, [sp]            // encoding: [0xe2,0x7f,0xc0,0x49]
// ERROR: instruction requires: lsui
  casplt     x0, x1, x2, x3, [x4]
// CHECK: casplt	x0, x1, x2, x3, [x4]            // encoding: [0x82,0xfc,0x80,0x49]
// ERROR: instruction requires: lsui
  casplt     x0, x1, x2, x3, [sp, #0]
// CHECK: casplt	x0, x1, x2, x3, [sp]            // encoding: [0xe2,0xff,0x80,0x49]
// ERROR: instruction requires: lsui
  caspalt    x0, x1, x2, x3, [x4]
// CHECK: caspalt	x0, x1, x2, x3, [x4]            // encoding: [0x82,0xfc,0xc0,0x49]
// ERROR: instruction requires: lsui
  caspalt    x0, x1, x2, x3, [sp, #0]
// CHECK: caspalt	x0, x1, x2, x3, [sp]            // encoding: [0xe2,0xff,0xc0,0x49]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// SWP(A|L)T instructions
//------------------------------------------------------------------------------
  swpt       w7, wzr, [x5]
// CHECK: swpt	w7, wzr, [x5]                   // encoding: [0xbf,0x84,0x27,0x19]
// ERROR: instruction requires: lsui
  swpt       x9, xzr, [sp]
// CHECK: swpt	x9, xzr, [sp]                   // encoding: [0xff,0x87,0x29,0x59]
// ERROR: instruction requires: lsui

  swpta      w7, wzr, [x5]
// CHECK: swpta	w7, wzr, [x5]                   // encoding: [0xbf,0x84,0xa7,0x19]
// ERROR: instruction requires: lsui
  swpta      x9, xzr, [sp]
// CHECK: swpta	x9, xzr, [sp]                   // encoding: [0xff,0x87,0xa9,0x59]
// ERROR: instruction requires: lsui

  swptl      w7, wzr, [x5]
// CHECK: swptl	w7, wzr, [x5]                   // encoding: [0xbf,0x84,0x67,0x19]
// ERROR: instruction requires: lsui
  swptl      x9, xzr, [sp]
// CHECK: swptl	x9, xzr, [sp]                   // encoding: [0xff,0x87,0x69,0x59]
// ERROR: instruction requires: lsui

  swptal     w7, wzr, [x5]
// CHECK: swptal	w7, wzr, [x5]                   // encoding: [0xbf,0x84,0xe7,0x19]
// ERROR: instruction requires: lsui
  swptal     x9, xzr, [sp]
// CHECK: swptal	x9, xzr, [sp]                   // encoding: [0xff,0x87,0xe9,0x59]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// LD{ADD|CLR|SET)(A|L|AL)T instructions
//------------------------------------------------------------------------------

  ldtadd     w7, wzr, [x5]
// CHECK: ldtadd	w7, wzr, [x5]                   // encoding: [0xbf,0x04,0x27,0x19]
// ERROR: instruction requires: lsui
  ldtadd     x9, xzr, [sp]
// CHECK: ldtadd	x9, xzr, [sp]                   // encoding: [0xff,0x07,0x29,0x59]
// ERROR: instruction requires: lsui

  ldtadda    w7, wzr, [x5]
// CHECK: ldtadda	w7, wzr, [x5]                   // encoding: [0xbf,0x04,0xa7,0x19]
// ERROR: instruction requires: lsui
  ldtadda    x9, xzr, [sp]
// CHECK: ldtadda	x9, xzr, [sp]                   // encoding: [0xff,0x07,0xa9,0x59]
// ERROR: instruction requires: lsui

  ldtaddl    w7, wzr, [x5]
// CHECK: ldtaddl	w7, wzr, [x5]                   // encoding: [0xbf,0x04,0x67,0x19]
// ERROR: instruction requires: lsui
  ldtaddl    x9, xzr, [sp]
// CHECK: ldtaddl	x9, xzr, [sp]                   // encoding: [0xff,0x07,0x69,0x59]
// ERROR: instruction requires: lsui

  ldtaddal   w7, wzr, [x5]
// CHECK: ldtaddal	w7, wzr, [x5]                   // encoding: [0xbf,0x04,0xe7,0x19]
// ERROR: instruction requires: lsui
  ldtaddal   x9, xzr, [sp]
// CHECK: ldtaddal	x9, xzr, [sp]                   // encoding: [0xff,0x07,0xe9,0x59]
// ERROR: instruction requires: lsui

  ldtclr     w7, wzr, [x5]
// CHECK: ldtclr	w7, wzr, [x5]                   // encoding: [0xbf,0x14,0x27,0x19]
// ERROR: instruction requires: lsui
  ldtclr     x9, xzr, [sp]
// CHECK: ldtclr	x9, xzr, [sp]                   // encoding: [0xff,0x17,0x29,0x59]
// ERROR: instruction requires: lsui

  ldtclrl    w7, wzr, [x5]
// CHECK: ldtclrl	w7, wzr, [x5]                   // encoding: [0xbf,0x14,0x67,0x19]
// ERROR: instruction requires: lsui
  ldtclrl    x9, xzr, [sp]
// CHECK: ldtclrl	x9, xzr, [sp]                   // encoding: [0xff,0x17,0x69,0x59]
// ERROR: instruction requires: lsui

  ldtclra    w7, wzr, [x5]
// CHECK: ldtclra	w7, wzr, [x5]                   // encoding: [0xbf,0x14,0xa7,0x19]
// ERROR: instruction requires: lsui
  ldtclra    x9, xzr, [sp]
// CHECK: ldtclra	x9, xzr, [sp]                   // encoding: [0xff,0x17,0xa9,0x59]
// ERROR: instruction requires: lsui

  ldtclral   w7, wzr, [x5]
// CHECK: ldtclral	w7, wzr, [x5]                   // encoding: [0xbf,0x14,0xe7,0x19]
// ERROR: instruction requires: lsui
  ldtclral   x9, xzr, [sp]
// CHECK: ldtclral	x9, xzr, [sp]                   // encoding: [0xff,0x17,0xe9,0x59]
// ERROR: instruction requires: lsui

  ldtset     w7, wzr, [x5]
// CHECK: ldtset	w7, wzr, [x5]                   // encoding: [0xbf,0x34,0x27,0x19]
// ERROR: instruction requires: lsui
  ldtset     x9, xzr, [sp]
// CHECK: ldtset	x9, xzr, [sp]                   // encoding: [0xff,0x37,0x29,0x59]
// ERROR: instruction requires: lsui

  ldtsetl    w7, wzr, [x5]
// CHECK: ldtsetl	w7, wzr, [x5]                   // encoding: [0xbf,0x34,0x67,0x19]
// ERROR: instruction requires: lsui
  ldtsetl    x9, xzr, [sp]
// CHECK: ldtsetl	x9, xzr, [sp]                   // encoding: [0xff,0x37,0x69,0x59]
// ERROR: instruction requires: lsui

  ldtseta    w7, wzr, [x5]
// CHECK: ldtseta	w7, wzr, [x5]                   // encoding: [0xbf,0x34,0xa7,0x19]
// ERROR: instruction requires: lsui
  ldtseta    x9, xzr, [sp]
// CHECK: ldtseta	x9, xzr, [sp]                   // encoding: [0xff,0x37,0xa9,0x59]
// ERROR: instruction requires: lsui

  ldtsetal   w7, wzr, [x5]
// CHECK: ldtsetal	w7, wzr, [x5]                   // encoding: [0xbf,0x34,0xe7,0x19]
// ERROR: instruction requires: lsui
  ldtsetal   x9, xzr, [sp]
// CHECK: ldtsetal	x9, xzr, [sp]                   // encoding: [0xff,0x37,0xe9,0x59]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// ST{ADD|CLR|SET)(A|L|AL)T instructions
//------------------------------------------------------------------------------

  sttadd     w0, [x2]
// CHECK: ldtadd	w0, wzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x19]
// ERROR: instruction requires: lsui
  sttadd     w2, [sp]
// CHECK: ldtadd	w2, wzr, [sp]                   // encoding: [0xff,0x07,0x22,0x19]
// ERROR: instruction requires: lsui
  sttadd     x0, [x2]
// CHECK: ldtadd	x0, xzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x59]
// ERROR: instruction requires: lsui
  sttadd     x2, [sp]
// CHECK: ldtadd	x2, xzr, [sp]                   // encoding: [0xff,0x07,0x22,0x59]
// ERROR: instruction requires: lsui

  sttaddl    w0, [x2]
// CHECK: ldtadd	w0, wzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x19]
// ERROR: instruction requires: lsui
  sttaddl    w2, [sp]
// CHECK: ldtadd	w2, wzr, [sp]                   // encoding: [0xff,0x07,0x22,0x19]
// ERROR: instruction requires: lsui
  sttaddl    x0, [x2]
// CHECK: ldtadd	x0, xzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x59]
// ERROR: instruction requires: lsui
  sttaddl    x2, [sp]
// CHECK: ldtadd	x2, xzr, [sp]                   // encoding: [0xff,0x07,0x22,0x59]
// ERROR: instruction requires: lsui

  sttadda    w0, [x2]
// CHECK: ldtadd	w0, wzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x19]
// ERROR: instruction requires: lsui
  sttadda    w2, [sp]
// CHECK: ldtadd	w2, wzr, [sp]                   // encoding: [0xff,0x07,0x22,0x19]
// ERROR: instruction requires: lsui
  sttadda    x0, [x2]
// CHECK: ldtadd	x0, xzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x59]
// ERROR: instruction requires: lsui
  sttadda    x2, [sp]
// CHECK: ldtadd	x2, xzr, [sp]                   // encoding: [0xff,0x07,0x22,0x59]
// ERROR: instruction requires: lsui

  sttaddal   w0, [x2]
// CHECK: ldtadd	w0, wzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x19]
// ERROR: instruction requires: lsui
  sttaddal   w2, [sp]
// CHECK: ldtadd	w2, wzr, [sp]                   // encoding: [0xff,0x07,0x22,0x19]
// ERROR: instruction requires: lsui
  sttaddal   x0, [x2]
// CHECK: ldtadd	x0, xzr, [x2]                   // encoding: [0x5f,0x04,0x20,0x59]
// ERROR: instruction requires: lsui
  sttaddal   x2, [sp]
// CHECK: ldtadd	x2, xzr, [sp]                   // encoding: [0xff,0x07,0x22,0x59]
// ERROR: instruction requires: lsui

  sttclr     w0, [x2]
// CHECK: ldtclr	w0, wzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x19]
// ERROR: instruction requires: lsui
  sttclr     w2, [sp]
// CHECK: ldtclr	w2, wzr, [sp]                   // encoding: [0xff,0x17,0x22,0x19]
// ERROR: instruction requires: lsui
  sttclr     x0, [x2]
// CHECK: ldtclr	x0, xzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x59]
// ERROR: instruction requires: lsui
  sttclr     x2, [sp]
// CHECK: ldtclr	x2, xzr, [sp]                   // encoding: [0xff,0x17,0x22,0x59]
// ERROR: instruction requires: lsui

  sttclra    w0, [x2]
// CHECK: ldtclr	w0, wzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x19]
// ERROR: instruction requires: lsui
  sttclra    w2, [sp]
// CHECK: ldtclr	w2, wzr, [sp]                   // encoding: [0xff,0x17,0x22,0x19]
// ERROR: instruction requires: lsui
  sttclra    x0, [x2]
// CHECK: ldtclr	x0, xzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x59]
// ERROR: instruction requires: lsui
  sttclra    x2, [sp]
// CHECK: ldtclr	x2, xzr, [sp]                   // encoding: [0xff,0x17,0x22,0x59]
// ERROR: instruction requires: lsui

  sttclrl    w0, [x2]
// CHECK: ldtclr	w0, wzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x19]
// ERROR: instruction requires: lsui
  sttclrl    w2, [sp]
// CHECK: ldtclr	w2, wzr, [sp]                   // encoding: [0xff,0x17,0x22,0x19]
// ERROR: instruction requires: lsui
  sttclrl    x0, [x2]
// CHECK: ldtclr	x0, xzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x59]
// ERROR: instruction requires: lsui
  sttclrl    x2, [sp]
// CHECK: ldtclr	x2, xzr, [sp]                   // encoding: [0xff,0x17,0x22,0x59]
// ERROR: instruction requires: lsui

  sttclral   w0, [x2]
// CHECK: ldtclr	w0, wzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x19]
// ERROR: instruction requires: lsui
  sttclral   x2, [sp]
// CHECK: ldtclr	x2, xzr, [sp]                   // encoding: [0xff,0x17,0x22,0x59]
// ERROR: instruction requires: lsui
  sttclral   x0, [x2]
// CHECK: ldtclr	x0, xzr, [x2]                   // encoding: [0x5f,0x14,0x20,0x59]
// ERROR: instruction requires: lsui
  sttclral   x2, [sp]
// CHECK: ldtclr	x2, xzr, [sp]                   // encoding: [0xff,0x17,0x22,0x59]
// ERROR: instruction requires: lsui

  sttset     w0, [x2]
// CHECK: ldtset	w0, wzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x19]
// ERROR: instruction requires: lsui
  sttset     w2, [sp]
// CHECK: ldtset	w2, wzr, [sp]                   // encoding: [0xff,0x37,0x22,0x19]
// ERROR: instruction requires: lsui
  sttset     x0, [x2]
// CHECK: ldtset	x0, xzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x59]
// ERROR: instruction requires: lsui
  sttset     x2, [sp]
// CHECK: ldtset	x2, xzr, [sp]                   // encoding: [0xff,0x37,0x22,0x59]
// ERROR: instruction requires: lsui

  sttseta    w0, [x2]
// CHECK: ldtset	w0, wzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x19]
// ERROR: instruction requires: lsui
  sttseta    w2, [sp]
// CHECK: ldtset	w2, wzr, [sp]                   // encoding: [0xff,0x37,0x22,0x19]
// ERROR: instruction requires: lsui
  sttseta    x0, [x2]
// CHECK: ldtset	x0, xzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x59]
// ERROR: instruction requires: lsui
  sttseta    x2, [sp]
// CHECK: ldtset	x2, xzr, [sp]                   // encoding: [0xff,0x37,0x22,0x59]
// ERROR: instruction requires: lsui

  sttsetl    w0, [x2]
// CHECK: ldtset	w0, wzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x19]
// ERROR: instruction requires: lsui
  sttsetl    w2, [sp]
// CHECK: ldtset	w2, wzr, [sp]                   // encoding: [0xff,0x37,0x22,0x19]
// ERROR: instruction requires: lsui
  sttsetl    x0, [x2]
// CHECK: ldtset	x0, xzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x59]
// ERROR: instruction requires: lsui
  sttsetl    x2, [sp]
// CHECK: ldtset	x2, xzr, [sp]                   // encoding: [0xff,0x37,0x22,0x59]
// ERROR: instruction requires: lsui

  sttsetal   w0, [x2]
// CHECK: ldtset	w0, wzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x19]
// ERROR: instruction requires: lsui
  sttsetal   x2, [sp]
// CHECK: ldtset	x2, xzr, [sp]                   // encoding: [0xff,0x37,0x22,0x59]
// ERROR: instruction requires: lsui
  sttsetal   x0, [x2]
// CHECK: ldtset	x0, xzr, [x2]                   // encoding: [0x5f,0x34,0x20,0x59]
// ERROR: instruction requires: lsui
  sttsetal   x2, [sp]
// CHECK: ldtset	x2, xzr, [sp]                   // encoding: [0xff,0x37,0x22,0x59]
// ERROR: instruction requires: lsui

//------------------------------------------------------------------------------
// Load/store non-temporal register pair (offset)
//------------------------------------------------------------------------------
  ldtnp      x21, x29, [x2, #504]
// CHECK: ldtnp	x21, x29, [x2, #504]            // encoding: [0x55,0xf4,0x5f,0xe8]
// ERROR: instruction requires: lsui
  ldtnp      x22, x23, [x3, #-512]
// CHECK: ldtnp	x22, x23, [x3, #-512]           // encoding: [0x76,0x5c,0x60,0xe8]
// ERROR: instruction requires: lsui
  ldtnp      x24, x25, [x4, #8]
// CHECK: ldtnp	x24, x25, [x4, #8]              // encoding: [0x98,0xe4,0x40,0xe8]
// ERROR: instruction requires: lsui
  ldtnp      q23, q29, [x1, #-1024]
// CHECK: ldtnp	q23, q29, [x1, #-1024]          // encoding: [0x37,0x74,0x60,0xec]
// ERROR: instruction requires: lsui

  sttnp      x3, x5, [sp]
// CHECK: sttnp	x3, x5, [sp]                    // encoding: [0xe3,0x17,0x00,0xe8]
// ERROR: instruction requires: lsui
  sttnp      x17, x19, [sp, #64]
// CHECK: sttnp	x17, x19, [sp, #64]             // encoding: [0xf1,0x4f,0x04,0xe8]
// ERROR: instruction requires: lsui
  sttnp      q3, q5, [sp]
// CHECK: sttnp	q3, q5, [sp]                    // encoding: [0xe3,0x17,0x00,0xec]
// ERROR: instruction requires: lsui
  sttnp      q17, q19, [sp, #1008]
// CHECK: sttnp	q17, q19, [sp, #1008]           // encoding: [0xf1,0xcf,0x1f,0xec]
// ERROR: instruction requires: lsui

