// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.1a,+lse -show-encoding < %s 2> %t | FileCheck %s
// RUN: FileCheck -check-prefix=CHECK-ERROR < %t %s
  .text

  cas w0, w1, [x2]
  cas w2, w3, [sp]
  casa w0, w1, [x2]
  casa w2, w3, [sp]
  casl w0, w1, [x2]
  casl w2, w3, [sp]
  casal w0, w1, [x2]
  casal w2, w3, [sp]
  // CHECK: cas w0, w1, [x2]      // encoding: [0x41,0x7c,0xa0,0x88]
  // CHECK: cas w2, w3, [sp]      // encoding: [0xe3,0x7f,0xa2,0x88]
  // CHECK: casa w0, w1, [x2]     // encoding: [0x41,0x7c,0xe0,0x88]
  // CHECK: casa w2, w3, [sp]     // encoding: [0xe3,0x7f,0xe2,0x88]
  // CHECK: casl w0, w1, [x2]     // encoding: [0x41,0xfc,0xa0,0x88]
  // CHECK: casl w2, w3, [sp]     // encoding: [0xe3,0xff,0xa2,0x88]
  // CHECK: casal w0, w1, [x2]    // encoding: [0x41,0xfc,0xe0,0x88]
  // CHECK: casal w2, w3, [sp]    // encoding: [0xe3,0xff,0xe2,0x88]

  casb w0, w1, [x2]
  casb w2, w3, [sp]
  cash w0, w1, [x2]
  cash w2, w3, [sp]
  casab w0, w1, [x2]
  casab w2, w3, [sp]
  caslb w0, w1, [x2]
  caslb w2, w3, [sp]
  // CHECK: casb w0, w1, [x2]     // encoding: [0x41,0x7c,0xa0,0x08]
  // CHECK: casb w2, w3, [sp]     // encoding: [0xe3,0x7f,0xa2,0x08]
  // CHECK: cash w0, w1, [x2]     // encoding: [0x41,0x7c,0xa0,0x48]
  // CHECK: cash w2, w3, [sp]     // encoding: [0xe3,0x7f,0xa2,0x48]
  // CHECK: casab w0, w1, [x2]    // encoding: [0x41,0x7c,0xe0,0x08]
  // CHECK: casab w2, w3, [sp]    // encoding: [0xe3,0x7f,0xe2,0x08]
  // CHECK: caslb w0, w1, [x2]    // encoding: [0x41,0xfc,0xa0,0x08]
  // CHECK: caslb w2, w3, [sp]    // encoding: [0xe3,0xff,0xa2,0x08]

  casalb w0, w1, [x2]
  casalb w2, w3, [sp]
  casah w0, w1, [x2]
  casah w2, w3, [sp]
  caslh w0, w1, [x2]
  caslh w2, w3, [sp]
  casalh w0, w1, [x2]
  casalh w2, w3, [sp]
  // CHECK: casalb w0, w1, [x2]   // encoding: [0x41,0xfc,0xe0,0x08]
  // CHECK: casalb w2, w3, [sp]   // encoding: [0xe3,0xff,0xe2,0x08]
  // CHECK: casah w0, w1, [x2]    // encoding: [0x41,0x7c,0xe0,0x48]
  // CHECK: casah w2, w3, [sp]    // encoding: [0xe3,0x7f,0xe2,0x48]
  // CHECK: caslh w0, w1, [x2]    // encoding: [0x41,0xfc,0xa0,0x48]
  // CHECK: caslh w2, w3, [sp]    // encoding: [0xe3,0xff,0xa2,0x48]
  // CHECK: casalh w0, w1, [x2]   // encoding: [0x41,0xfc,0xe0,0x48]
  // CHECK: casalh w2, w3, [sp]   // encoding: [0xe3,0xff,0xe2,0x48]

  cas x0, x1, [x2]
  cas x2, x3, [sp]
  casa x0, x1, [x2]
  casa x2, x3, [sp]
  casl x0, x1, [x2]
  casl x2, x3, [sp]
  casal x0, x1, [x2]
  casal x2, x3, [sp]
  // CHECK: cas x0, x1, [x2]      // encoding: [0x41,0x7c,0xa0,0xc8]
  // CHECK: cas x2, x3, [sp]      // encoding: [0xe3,0x7f,0xa2,0xc8]
  // CHECK: casa x0, x1, [x2]     // encoding: [0x41,0x7c,0xe0,0xc8]
  // CHECK: casa x2, x3, [sp]     // encoding: [0xe3,0x7f,0xe2,0xc8]
  // CHECK: casl x0, x1, [x2]     // encoding: [0x41,0xfc,0xa0,0xc8]
  // CHECK: casl x2, x3, [sp]     // encoding: [0xe3,0xff,0xa2,0xc8]
  // CHECK: casal x0, x1, [x2]    // encoding: [0x41,0xfc,0xe0,0xc8]
  // CHECK: casal x2, x3, [sp]    // encoding: [0xe3,0xff,0xe2,0xc8]

  swp w0, w1, [x2]
  swp w2, w3, [sp]
  swpa w0, w1, [x2]
  swpa w2, w3, [sp]
  swpl w0, w1, [x2]
  swpl w2, w3, [sp]
  swpal w0, w1, [x2]
  swpal w2, w3, [sp]
  // CHECK: swp w0, w1, [x2]      // encoding: [0x41,0x80,0x20,0xb8]
  // CHECK: swp w2, w3, [sp]      // encoding: [0xe3,0x83,0x22,0xb8]
  // CHECK: swpa w0, w1, [x2]     // encoding: [0x41,0x80,0xa0,0xb8]
  // CHECK: swpa w2, w3, [sp]     // encoding: [0xe3,0x83,0xa2,0xb8]
  // CHECK: swpl w0, w1, [x2]     // encoding: [0x41,0x80,0x60,0xb8]
  // CHECK: swpl w2, w3, [sp]     // encoding: [0xe3,0x83,0x62,0xb8]
  // CHECK: swpal w0, w1, [x2]    // encoding: [0x41,0x80,0xe0,0xb8]
  // CHECK: swpal w2, w3, [sp]    // encoding: [0xe3,0x83,0xe2,0xb8]

  swpb w0, w1, [x2]
  swpb w2, w3, [sp]
  swph w0, w1, [x2]
  swph w2, w3, [sp]
  swpab w0, w1, [x2]
  swpab w2, w3, [sp]
  swplb w0, w1, [x2]
  swplb w2, w3, [sp]
  // CHECK: swpb w0, w1, [x2]     // encoding: [0x41,0x80,0x20,0x38]
  // CHECK: swpb w2, w3, [sp]     // encoding: [0xe3,0x83,0x22,0x38]
  // CHECK: swph w0, w1, [x2]     // encoding: [0x41,0x80,0x20,0x78]
  // CHECK: swph w2, w3, [sp]     // encoding: [0xe3,0x83,0x22,0x78]
  // CHECK: swpab w0, w1, [x2]    // encoding: [0x41,0x80,0xa0,0x38]
  // CHECK: swpab w2, w3, [sp]    // encoding: [0xe3,0x83,0xa2,0x38]
  // CHECK: swplb w0, w1, [x2]    // encoding: [0x41,0x80,0x60,0x38]
  // CHECK: swplb w2, w3, [sp]    // encoding: [0xe3,0x83,0x62,0x38]

  swpalb w0, w1, [x2]
  swpalb w2, w3, [sp]
  swpah w0, w1, [x2]
  swpah w2, w3, [sp]
  swplh w0, w1, [x2]
  swplh w2, w3, [sp]
  swpalh w0, w1, [x2]
  swpalh w2, w3, [sp]
  // CHECK: swpalb w0, w1, [x2]   // encoding: [0x41,0x80,0xe0,0x38]
  // CHECK: swpalb w2, w3, [sp]   // encoding: [0xe3,0x83,0xe2,0x38]
  // CHECK: swpah w0, w1, [x2]    // encoding: [0x41,0x80,0xa0,0x78]
  // CHECK: swpah w2, w3, [sp]    // encoding: [0xe3,0x83,0xa2,0x78]
  // CHECK: swplh w0, w1, [x2]    // encoding: [0x41,0x80,0x60,0x78]
  // CHECK: swplh w2, w3, [sp]    // encoding: [0xe3,0x83,0x62,0x78]
  // CHECK: swpalh w0, w1, [x2]   // encoding: [0x41,0x80,0xe0,0x78]
  // CHECK: swpalh w2, w3, [sp]   // encoding: [0xe3,0x83,0xe2,0x78]

  swp x0, x1, [x2]
  swp x2, x3, [sp]
  swpa x0, x1, [x2]
  swpa x2, x3, [sp]
  swpl x0, x1, [x2]
  swpl x2, x3, [sp]
  swpal x0, x1, [x2]
  swpal x2, x3, [sp]
  // CHECK: swp x0, x1, [x2]      // encoding: [0x41,0x80,0x20,0xf8]
  // CHECK: swp x2, x3, [sp]      // encoding: [0xe3,0x83,0x22,0xf8]
  // CHECK: swpa x0, x1, [x2]     // encoding: [0x41,0x80,0xa0,0xf8]
  // CHECK: swpa x2, x3, [sp]     // encoding: [0xe3,0x83,0xa2,0xf8]
  // CHECK: swpl x0, x1, [x2]     // encoding: [0x41,0x80,0x60,0xf8]
  // CHECK: swpl x2, x3, [sp]     // encoding: [0xe3,0x83,0x62,0xf8]
  // CHECK: swpal x0, x1, [x2]    // encoding: [0x41,0x80,0xe0,0xf8]
  // CHECK: swpal x2, x3, [sp]    // encoding: [0xe3,0x83,0xe2,0xf8]

  casp w0, w1, w2, w3, [x5]
  casp w4, w5, w6, w7, [sp]
  casp x0, x1, x2, x3, [x2]
  casp x4, x5, x6, x7, [sp]
  caspa w0, w1, w2, w3, [x5]
  caspa w4, w5, w6, w7, [sp]
  caspa x0, x1, x2, x3, [x2]
  caspa x4, x5, x6, x7, [sp]
  // CHECK: casp w0, w1, w2, w3, [x5]     // encoding: [0xa2,0x7c,0x20,0x08]
  // CHECK: casp w4, w5, w6, w7, [sp]     // encoding: [0xe6,0x7f,0x24,0x08]
  // CHECK: casp x0, x1, x2, x3, [x2]     // encoding: [0x42,0x7c,0x20,0x48]
  // CHECK: casp x4, x5, x6, x7, [sp]     // encoding: [0xe6,0x7f,0x24,0x48]
  // CHECK: caspa w0, w1, w2, w3, [x5]    // encoding: [0xa2,0x7c,0x60,0x08]
  // CHECK: caspa w4, w5, w6, w7, [sp]    // encoding: [0xe6,0x7f,0x64,0x08]
  // CHECK: caspa x0, x1, x2, x3, [x2]    // encoding: [0x42,0x7c,0x60,0x48]
  // CHECK: caspa x4, x5, x6, x7, [sp]    // encoding: [0xe6,0x7f,0x64,0x48]

  caspl w0, w1, w2, w3, [x5]
  caspl w4, w5, w6, w7, [sp]
  caspl x0, x1, x2, x3, [x2]
  caspl x4, x5, x6, x7, [sp]
  caspal w0, w1, w2, w3, [x5]
  caspal w4, w5, w6, w7, [sp]
  caspal x0, x1, x2, x3, [x2]
  caspal x4, x5, x6, x7, [sp]
  // CHECK: caspl w0, w1, w2, w3, [x5]    // encoding: [0xa2,0xfc,0x20,0x08]
  // CHECK: caspl w4, w5, w6, w7, [sp]    // encoding: [0xe6,0xff,0x24,0x08]
  // CHECK: caspl x0, x1, x2, x3, [x2]    // encoding: [0x42,0xfc,0x20,0x48]
  // CHECK: caspl x4, x5, x6, x7, [sp]    // encoding: [0xe6,0xff,0x24,0x48]
  // CHECK: caspal w0, w1, w2, w3, [x5]   // encoding: [0xa2,0xfc,0x60,0x08]
  // CHECK: caspal w4, w5, w6, w7, [sp]   // encoding: [0xe6,0xff,0x64,0x08]
  // CHECK: caspal x0, x1, x2, x3, [x2]   // encoding: [0x42,0xfc,0x60,0x48]
  // CHECK: caspal x4, x5, x6, x7, [sp]   // encoding: [0xe6,0xff,0x64,0x48]

  ldadd w0, w1, [x2]
  ldadd w2, w3, [sp]
  ldadda w0, w1, [x2]
  ldadda w2, w3, [sp]
  ldaddl w0, w1, [x2]
  ldaddl w2, w3, [sp]
  ldaddal w0, w1, [x2]
  ldaddal w2, w3, [sp]
  // CHECK: ldadd w0, w1, [x2]     // encoding: [0x41,0x00,0x20,0xb8]
  // CHECK: ldadd w2, w3, [sp]     // encoding: [0xe3,0x03,0x22,0xb8]
  // CHECK: ldadda w0, w1, [x2]    // encoding: [0x41,0x00,0xa0,0xb8]
  // CHECK: ldadda w2, w3, [sp]    // encoding: [0xe3,0x03,0xa2,0xb8]
  // CHECK: ldaddl w0, w1, [x2]    // encoding: [0x41,0x00,0x60,0xb8]
  // CHECK: ldaddl w2, w3, [sp]    // encoding: [0xe3,0x03,0x62,0xb8]
  // CHECK: ldaddal w0, w1, [x2]   // encoding: [0x41,0x00,0xe0,0xb8]
  // CHECK: ldaddal w2, w3, [sp]   // encoding: [0xe3,0x03,0xe2,0xb8]

  ldaddb w0, w1, [x2]
  ldaddb w2, w3, [sp]
  ldaddh w0, w1, [x2]
  ldaddh w2, w3, [sp]
  ldaddab w0, w1, [x2]
  ldaddab w2, w3, [sp]
  ldaddlb w0, w1, [x2]
  ldaddlb w2, w3, [sp]
  // CHECK: ldaddb w0, w1, [x2]       // encoding: [0x41,0x00,0x20,0x38]
  // CHECK: ldaddb w2, w3, [sp]       // encoding: [0xe3,0x03,0x22,0x38]
  // CHECK: ldaddh w0, w1, [x2]       // encoding: [0x41,0x00,0x20,0x78]
  // CHECK: ldaddh w2, w3, [sp]       // encoding: [0xe3,0x03,0x22,0x78]
  // CHECK: ldaddab w0, w1, [x2]      // encoding: [0x41,0x00,0xa0,0x38]
  // CHECK: ldaddab w2, w3, [sp]      // encoding: [0xe3,0x03,0xa2,0x38]
  // CHECK: ldaddlb w0, w1, [x2]      // encoding: [0x41,0x00,0x60,0x38]
  // CHECK: ldaddlb w2, w3, [sp]      // encoding: [0xe3,0x03,0x62,0x38]

  ldaddalb w0, w1, [x2]
  ldaddalb w2, w3, [sp]
  ldaddah w0, w1, [x2]
  ldaddah w2, w3, [sp]
  ldaddlh w0, w1, [x2]
  ldaddlh w2, w3, [sp]
  ldaddalh w0, w1, [x2]
  ldaddalh w2, w3, [sp]
  // CHECK: ldaddalb w0, w1, [x2]   // encoding: [0x41,0x00,0xe0,0x38]
  // CHECK: ldaddalb w2, w3, [sp]   // encoding: [0xe3,0x03,0xe2,0x38]
  // CHECK: ldaddah w0, w1, [x2]    // encoding: [0x41,0x00,0xa0,0x78]
  // CHECK: ldaddah w2, w3, [sp]    // encoding: [0xe3,0x03,0xa2,0x78]
  // CHECK: ldaddlh w0, w1, [x2]    // encoding: [0x41,0x00,0x60,0x78]
  // CHECK: ldaddlh w2, w3, [sp]    // encoding: [0xe3,0x03,0x62,0x78]
  // CHECK: ldaddalh w0, w1, [x2]   // encoding: [0x41,0x00,0xe0,0x78]
  // CHECK: ldaddalh w2, w3, [sp]   // encoding: [0xe3,0x03,0xe2,0x78]

  ldadd x0, x1, [x2]
  ldadd x2, x3, [sp]
  ldadda x0, x1, [x2]
  ldadda x2, x3, [sp]
  ldaddl x0, x1, [x2]
  ldaddl x2, x3, [sp]
  ldaddal x0, x1, [x2]
  ldaddal x2, x3, [sp]
  // CHECK: ldadd x0, x1, [x2]    // encoding: [0x41,0x00,0x20,0xf8]
  // CHECK: ldadd x2, x3, [sp]    // encoding: [0xe3,0x03,0x22,0xf8]
  // CHECK: ldadda x0, x1, [x2]   // encoding: [0x41,0x00,0xa0,0xf8]
  // CHECK: ldadda x2, x3, [sp]   // encoding: [0xe3,0x03,0xa2,0xf8]
  // CHECK: ldaddl x0, x1, [x2]   // encoding: [0x41,0x00,0x60,0xf8]
  // CHECK: ldaddl x2, x3, [sp]   // encoding: [0xe3,0x03,0x62,0xf8]
  // CHECK: ldaddal x0, x1, [x2]  // encoding: [0x41,0x00,0xe0,0xf8]
  // CHECK: ldaddal x2, x3, [sp]  // encoding: [0xe3,0x03,0xe2,0xf8]

  ldclr w0, w1, [x2]
  ldclr w2, w3, [sp]
  ldclra w0, w1, [x2]
  ldclra w2, w3, [sp]
  ldclrl w0, w1, [x2]
  ldclrl w2, w3, [sp]
  ldclral w0, w1, [x2]
  ldclral w2, w3, [sp]
  // CHECK: ldclr w0, w1, [x2]    // encoding: [0x41,0x10,0x20,0xb8]
  // CHECK: ldclr w2, w3, [sp]    // encoding: [0xe3,0x13,0x22,0xb8]
  // CHECK: ldclra w0, w1, [x2]   // encoding: [0x41,0x10,0xa0,0xb8]
  // CHECK: ldclra w2, w3, [sp]   // encoding: [0xe3,0x13,0xa2,0xb8]
  // CHECK: ldclrl w0, w1, [x2]   // encoding: [0x41,0x10,0x60,0xb8]
  // CHECK: ldclrl w2, w3, [sp]   // encoding: [0xe3,0x13,0x62,0xb8]
  // CHECK: ldclral w0, w1, [x2]  // encoding: [0x41,0x10,0xe0,0xb8]
  // CHECK: ldclral w2, w3, [sp]  // encoding: [0xe3,0x13,0xe2,0xb8]

  ldclrb w0, w1, [x2]
  ldclrb w2, w3, [sp]
  ldclrh w0, w1, [x2]
  ldclrh w2, w3, [sp]
  ldclrab w0, w1, [x2]
  ldclrab w2, w3, [sp]
  ldclrlb w0, w1, [x2]
  ldclrlb w2, w3, [sp]
  // CHECK: ldclrb w0, w1, [x2]    // encoding: [0x41,0x10,0x20,0x38]
  // CHECK: ldclrb w2, w3, [sp]    // encoding: [0xe3,0x13,0x22,0x38]
  // CHECK: ldclrh w0, w1, [x2]    // encoding: [0x41,0x10,0x20,0x78]
  // CHECK: ldclrh w2, w3, [sp]    // encoding: [0xe3,0x13,0x22,0x78]
  // CHECK: ldclrab w0, w1, [x2]   // encoding: [0x41,0x10,0xa0,0x38]
  // CHECK: ldclrab w2, w3, [sp]   // encoding: [0xe3,0x13,0xa2,0x38]
  // CHECK: ldclrlb w0, w1, [x2]   // encoding: [0x41,0x10,0x60,0x38]
  // CHECK: ldclrlb w2, w3, [sp]   // encoding: [0xe3,0x13,0x62,0x38]

  ldclralb w0, w1, [x2]
  ldclralb w2, w3, [sp]
  ldclrah w0, w1, [x2]
  ldclrah w2, w3, [sp]
  ldclrlh w0, w1, [x2]
  ldclrlh w2, w3, [sp]
  ldclralh w0, w1, [x2]
  ldclralh w2, w3, [sp]
  // CHECK: ldclralb w0, w1, [x2]   // encoding: [0x41,0x10,0xe0,0x38]
  // CHECK: ldclralb w2, w3, [sp]   // encoding: [0xe3,0x13,0xe2,0x38]
  // CHECK: ldclrah w0, w1, [x2]    // encoding: [0x41,0x10,0xa0,0x78]
  // CHECK: ldclrah w2, w3, [sp]    // encoding: [0xe3,0x13,0xa2,0x78]
  // CHECK: ldclrlh w0, w1, [x2]    // encoding: [0x41,0x10,0x60,0x78]
  // CHECK: ldclrlh w2, w3, [sp]    // encoding: [0xe3,0x13,0x62,0x78]
  // CHECK: ldclralh w0, w1, [x2]   // encoding: [0x41,0x10,0xe0,0x78]
  // CHECK: ldclralh w2, w3, [sp]   // encoding: [0xe3,0x13,0xe2,0x78]

  ldclr x0, x1, [x2]
  ldclr x2, x3, [sp]
  ldclra x0, x1, [x2]
  ldclra x2, x3, [sp]
  ldclrl x0, x1, [x2]
  ldclrl x2, x3, [sp]
  ldclral x0, x1, [x2]
  ldclral x2, x3, [sp]
  // CHECK: ldclr x0, x1, [x2]      // encoding: [0x41,0x10,0x20,0xf8]
  // CHECK: ldclr x2, x3, [sp]      // encoding: [0xe3,0x13,0x22,0xf8]
  // CHECK: ldclra x0, x1, [x2]     // encoding: [0x41,0x10,0xa0,0xf8]
  // CHECK: ldclra x2, x3, [sp]     // encoding: [0xe3,0x13,0xa2,0xf8]
  // CHECK: ldclrl x0, x1, [x2]     // encoding: [0x41,0x10,0x60,0xf8]
  // CHECK: ldclrl x2, x3, [sp]     // encoding: [0xe3,0x13,0x62,0xf8]
  // CHECK: ldclral x0, x1, [x2]    // encoding: [0x41,0x10,0xe0,0xf8]
  // CHECK: ldclral x2, x3, [sp]    // encoding: [0xe3,0x13,0xe2,0xf8]

  ldeor w0, w1, [x2]
  ldeor w2, w3, [sp]
  ldeora w0, w1, [x2]
  ldeora w2, w3, [sp]
  ldeorl w0, w1, [x2]
  ldeorl w2, w3, [sp]
  ldeoral w0, w1, [x2]
  ldeoral w2, w3, [sp]
  // CHECK: ldeor w0, w1, [x2]    // encoding: [0x41,0x20,0x20,0xb8]
  // CHECK: ldeor w2, w3, [sp]    // encoding: [0xe3,0x23,0x22,0xb8]
  // CHECK: ldeora w0, w1, [x2]   // encoding: [0x41,0x20,0xa0,0xb8]
  // CHECK: ldeora w2, w3, [sp]   // encoding: [0xe3,0x23,0xa2,0xb8]
  // CHECK: ldeorl w0, w1, [x2]   // encoding: [0x41,0x20,0x60,0xb8]
  // CHECK: ldeorl w2, w3, [sp]   // encoding: [0xe3,0x23,0x62,0xb8]
  // CHECK: ldeoral w0, w1, [x2]  // encoding: [0x41,0x20,0xe0,0xb8]
  // CHECK: ldeoral w2, w3, [sp]  // encoding: [0xe3,0x23,0xe2,0xb8]

  ldeorb w0, w1, [x2]
  ldeorb w2, w3, [sp]
  ldeorh w0, w1, [x2]
  ldeorh w2, w3, [sp]
  ldeorab w0, w1, [x2]
  ldeorab w2, w3, [sp]
  ldeorlb w0, w1, [x2]
  ldeorlb w2, w3, [sp]
  // CHECK: ldeorb w0, w1, [x2]    // encoding: [0x41,0x20,0x20,0x38]
  // CHECK: ldeorb w2, w3, [sp]    // encoding: [0xe3,0x23,0x22,0x38]
  // CHECK: ldeorh w0, w1, [x2]    // encoding: [0x41,0x20,0x20,0x78]
  // CHECK: ldeorh w2, w3, [sp]    // encoding: [0xe3,0x23,0x22,0x78]
  // CHECK: ldeorab w0, w1, [x2]   // encoding: [0x41,0x20,0xa0,0x38]
  // CHECK: ldeorab w2, w3, [sp]   // encoding: [0xe3,0x23,0xa2,0x38]
  // CHECK: ldeorlb w0, w1, [x2]   // encoding: [0x41,0x20,0x60,0x38]
  // CHECK: ldeorlb w2, w3, [sp]   // encoding: [0xe3,0x23,0x62,0x38]

  ldeoralb w0, w1, [x2]
  ldeoralb w2, w3, [sp]
  ldeorah w0, w1, [x2]
  ldeorah w2, w3, [sp]
  ldeorlh w0, w1, [x2]
  ldeorlh w2, w3, [sp]
  ldeoralh w0, w1, [x2]
  ldeoralh w2, w3, [sp]
  // CHECK: ldeoralb w0, w1, [x2]   // encoding: [0x41,0x20,0xe0,0x38]
  // CHECK: ldeoralb w2, w3, [sp]   // encoding: [0xe3,0x23,0xe2,0x38]
  // CHECK: ldeorah w0, w1, [x2]    // encoding: [0x41,0x20,0xa0,0x78]
  // CHECK: ldeorah w2, w3, [sp]    // encoding: [0xe3,0x23,0xa2,0x78]
  // CHECK: ldeorlh w0, w1, [x2]    // encoding: [0x41,0x20,0x60,0x78]
  // CHECK: ldeorlh w2, w3, [sp]    // encoding: [0xe3,0x23,0x62,0x78]
  // CHECK: ldeoralh w0, w1, [x2]   // encoding: [0x41,0x20,0xe0,0x78]
  // CHECK: ldeoralh w2, w3, [sp]   // encoding: [0xe3,0x23,0xe2,0x78]

  ldeor x0, x1, [x2]
  ldeor x2, x3, [sp]
  ldeora x0, x1, [x2]
  ldeora x2, x3, [sp]
  ldeorl x0, x1, [x2]
  ldeorl x2, x3, [sp]
  ldeoral x0, x1, [x2]
  ldeoral x2, x3, [sp]
  // CHECK: ldeor x0, x1, [x2]     // encoding: [0x41,0x20,0x20,0xf8]
  // CHECK: ldeor x2, x3, [sp]     // encoding: [0xe3,0x23,0x22,0xf8]
  // CHECK: ldeora x0, x1, [x2]    // encoding: [0x41,0x20,0xa0,0xf8]
  // CHECK: ldeora x2, x3, [sp]    // encoding: [0xe3,0x23,0xa2,0xf8]
  // CHECK: ldeorl x0, x1, [x2]    // encoding: [0x41,0x20,0x60,0xf8]
  // CHECK: ldeorl x2, x3, [sp]    // encoding: [0xe3,0x23,0x62,0xf8]
  // CHECK: ldeoral x0, x1, [x2]   // encoding: [0x41,0x20,0xe0,0xf8]
  // CHECK: ldeoral x2, x3, [sp]   // encoding: [0xe3,0x23,0xe2,0xf8]

  ldset w0, w1, [x2]
  ldset w2, w3, [sp]
  ldseta w0, w1, [x2]
  ldseta w2, w3, [sp]
  ldsetl w0, w1, [x2]
  ldsetl w2, w3, [sp]
  ldsetal w0, w1, [x2]
  ldsetal w2, w3, [sp]
  // CHECK: ldset w0, w1, [x2]      // encoding: [0x41,0x30,0x20,0xb8]
  // CHECK: ldset w2, w3, [sp]      // encoding: [0xe3,0x33,0x22,0xb8]
  // CHECK: ldseta w0, w1, [x2]     // encoding: [0x41,0x30,0xa0,0xb8]
  // CHECK: ldseta w2, w3, [sp]     // encoding: [0xe3,0x33,0xa2,0xb8]
  // CHECK: ldsetl w0, w1, [x2]     // encoding: [0x41,0x30,0x60,0xb8]
  // CHECK: ldsetl w2, w3, [sp]     // encoding: [0xe3,0x33,0x62,0xb8]
  // CHECK: ldsetal w0, w1, [x2]    // encoding: [0x41,0x30,0xe0,0xb8]
  // CHECK: ldsetal w2, w3, [sp]    // encoding: [0xe3,0x33,0xe2,0xb8]

  ldsetb w0, w1, [x2]
  ldsetb w2, w3, [sp]
  ldseth w0, w1, [x2]
  ldseth w2, w3, [sp]
  ldsetab w0, w1, [x2]
  ldsetab w2, w3, [sp]
  ldsetlb w0, w1, [x2]
  ldsetlb w2, w3, [sp]
  // CHECK: ldsetb w0, w1, [x2]     // encoding: [0x41,0x30,0x20,0x38]
  // CHECK: ldsetb w2, w3, [sp]     // encoding: [0xe3,0x33,0x22,0x38]
  // CHECK: ldseth w0, w1, [x2]     // encoding: [0x41,0x30,0x20,0x78]
  // CHECK: ldseth w2, w3, [sp]     // encoding: [0xe3,0x33,0x22,0x78]
  // CHECK: ldsetab w0, w1, [x2]    // encoding: [0x41,0x30,0xa0,0x38]
  // CHECK: ldsetab w2, w3, [sp]    // encoding: [0xe3,0x33,0xa2,0x38]
  // CHECK: ldsetlb w0, w1, [x2]    // encoding: [0x41,0x30,0x60,0x38]
  // CHECK: ldsetlb w2, w3, [sp]    // encoding: [0xe3,0x33,0x62,0x38]

  ldsetalb w0, w1, [x2]
  ldsetalb w2, w3, [sp]
  ldsetah w0, w1, [x2]
  ldsetah w2, w3, [sp]
  ldsetlh w0, w1, [x2]
  ldsetlh w2, w3, [sp]
  ldsetalh w0, w1, [x2]
  ldsetalh w2, w3, [sp]
  // CHECK: ldsetalb w0, w1, [x2]     // encoding: [0x41,0x30,0xe0,0x38]
  // CHECK: ldsetalb w2, w3, [sp]     // encoding: [0xe3,0x33,0xe2,0x38]
  // CHECK: ldsetah w0, w1, [x2]      // encoding: [0x41,0x30,0xa0,0x78]
  // CHECK: ldsetah w2, w3, [sp]      // encoding: [0xe3,0x33,0xa2,0x78]
  // CHECK: ldsetlh w0, w1, [x2]      // encoding: [0x41,0x30,0x60,0x78]
  // CHECK: ldsetlh w2, w3, [sp]      // encoding: [0xe3,0x33,0x62,0x78]
  // CHECK: ldsetalh w0, w1, [x2]     // encoding: [0x41,0x30,0xe0,0x78]
  // CHECK: ldsetalh w2, w3, [sp]     // encoding: [0xe3,0x33,0xe2,0x78]

  ldset x0, x1, [x2]
  ldset x2, x3, [sp]
  ldseta x0, x1, [x2]
  ldseta x2, x3, [sp]
  ldsetl x0, x1, [x2]
  ldsetl x2, x3, [sp]
  ldsetal x0, x1, [x2]
  ldsetal x2, x3, [sp]
  // CHECK: ldset x0, x1, [x2]     // encoding: [0x41,0x30,0x20,0xf8]
  // CHECK: ldset x2, x3, [sp]     // encoding: [0xe3,0x33,0x22,0xf8]
  // CHECK: ldseta x0, x1, [x2]    // encoding: [0x41,0x30,0xa0,0xf8]
  // CHECK: ldseta x2, x3, [sp]    // encoding: [0xe3,0x33,0xa2,0xf8]
  // CHECK: ldsetl x0, x1, [x2]    // encoding: [0x41,0x30,0x60,0xf8]
  // CHECK: ldsetl x2, x3, [sp]    // encoding: [0xe3,0x33,0x62,0xf8]
  // CHECK: ldsetal x0, x1, [x2]   // encoding: [0x41,0x30,0xe0,0xf8]
  // CHECK: ldsetal x2, x3, [sp]   // encoding: [0xe3,0x33,0xe2,0xf8]

  ldsmax w0, w1, [x2]
  ldsmax w2, w3, [sp]
  ldsmaxa w0, w1, [x2]
  ldsmaxa w2, w3, [sp]
  ldsmaxl w0, w1, [x2]
  ldsmaxl w2, w3, [sp]
  ldsmaxal w0, w1, [x2]
  ldsmaxal w2, w3, [sp]
  // CHECK: ldsmax w0, w1, [x2]     // encoding: [0x41,0x40,0x20,0xb8]
  // CHECK: ldsmax w2, w3, [sp]     // encoding: [0xe3,0x43,0x22,0xb8]
  // CHECK: ldsmaxa w0, w1, [x2]    // encoding: [0x41,0x40,0xa0,0xb8]
  // CHECK: ldsmaxa w2, w3, [sp]    // encoding: [0xe3,0x43,0xa2,0xb8]
  // CHECK: ldsmaxl w0, w1, [x2]    // encoding: [0x41,0x40,0x60,0xb8]
  // CHECK: ldsmaxl w2, w3, [sp]    // encoding: [0xe3,0x43,0x62,0xb8]
  // CHECK: ldsmaxal w0, w1, [x2]   // encoding: [0x41,0x40,0xe0,0xb8]
  // CHECK: ldsmaxal w2, w3, [sp]   // encoding: [0xe3,0x43,0xe2,0xb8]

  ldsmaxb w0, w1, [x2]
  ldsmaxb w2, w3, [sp]
  ldsmaxh w0, w1, [x2]
  ldsmaxh w2, w3, [sp]
  ldsmaxab w0, w1, [x2]
  ldsmaxab w2, w3, [sp]
  ldsmaxlb w0, w1, [x2]
  ldsmaxlb w2, w3, [sp]
  // CHECK: ldsmaxb w0, w1, [x2]     // encoding: [0x41,0x40,0x20,0x38]
  // CHECK: ldsmaxb w2, w3, [sp]     // encoding: [0xe3,0x43,0x22,0x38]
  // CHECK: ldsmaxh w0, w1, [x2]     // encoding: [0x41,0x40,0x20,0x78]
  // CHECK: ldsmaxh w2, w3, [sp]     // encoding: [0xe3,0x43,0x22,0x78]
  // CHECK: ldsmaxab w0, w1, [x2]    // encoding: [0x41,0x40,0xa0,0x38]
  // CHECK: ldsmaxab w2, w3, [sp]    // encoding: [0xe3,0x43,0xa2,0x38]
  // CHECK: ldsmaxlb w0, w1, [x2]    // encoding: [0x41,0x40,0x60,0x38]
  // CHECK: ldsmaxlb w2, w3, [sp]    // encoding: [0xe3,0x43,0x62,0x38]

  ldsmaxalb w0, w1, [x2]
  ldsmaxalb w2, w3, [sp]
  ldsmaxah w0, w1, [x2]
  ldsmaxah w2, w3, [sp]
  ldsmaxlh w0, w1, [x2]
  ldsmaxlh w2, w3, [sp]
  ldsmaxalh w0, w1, [x2]
  ldsmaxalh w2, w3, [sp]
  // CHECK: ldsmaxalb w0, w1, [x2]    // encoding: [0x41,0x40,0xe0,0x38]
  // CHECK: ldsmaxalb w2, w3, [sp]    // encoding: [0xe3,0x43,0xe2,0x38]
  // CHECK: ldsmaxah w0, w1, [x2]     // encoding: [0x41,0x40,0xa0,0x78]
  // CHECK: ldsmaxah w2, w3, [sp]     // encoding: [0xe3,0x43,0xa2,0x78]
  // CHECK: ldsmaxlh w0, w1, [x2]     // encoding: [0x41,0x40,0x60,0x78]
  // CHECK: ldsmaxlh w2, w3, [sp]     // encoding: [0xe3,0x43,0x62,0x78]
  // CHECK: ldsmaxalh w0, w1, [x2]    // encoding: [0x41,0x40,0xe0,0x78]
  // CHECK: ldsmaxalh w2, w3, [sp]    // encoding: [0xe3,0x43,0xe2,0x78]

  ldsmax x0, x1, [x2]
  ldsmax x2, x3, [sp]
  ldsmaxa x0, x1, [x2]
  ldsmaxa x2, x3, [sp]
  ldsmaxl x0, x1, [x2]
  ldsmaxl x2, x3, [sp]
  ldsmaxal x0, x1, [x2]
  ldsmaxal x2, x3, [sp]
  // CHECK: ldsmax x0, x1, [x2]     // encoding: [0x41,0x40,0x20,0xf8]
  // CHECK: ldsmax x2, x3, [sp]     // encoding: [0xe3,0x43,0x22,0xf8]
  // CHECK: ldsmaxa x0, x1, [x2]    // encoding: [0x41,0x40,0xa0,0xf8]
  // CHECK: ldsmaxa x2, x3, [sp]    // encoding: [0xe3,0x43,0xa2,0xf8]
  // CHECK: ldsmaxl x0, x1, [x2]    // encoding: [0x41,0x40,0x60,0xf8]
  // CHECK: ldsmaxl x2, x3, [sp]    // encoding: [0xe3,0x43,0x62,0xf8]
  // CHECK: ldsmaxal x0, x1, [x2]   // encoding: [0x41,0x40,0xe0,0xf8]
  // CHECK: ldsmaxal x2, x3, [sp]   // encoding: [0xe3,0x43,0xe2,0xf8]

  ldsmin w0, w1, [x2]
  ldsmin w2, w3, [sp]
  ldsmina w0, w1, [x2]
  ldsmina w2, w3, [sp]
  ldsminl w0, w1, [x2]
  ldsminl w2, w3, [sp]
  ldsminal w0, w1, [x2]
  ldsminal w2, w3, [sp]
  // CHECK: ldsmin w0, w1, [x2]     // encoding: [0x41,0x50,0x20,0xb8]
  // CHECK: ldsmin w2, w3, [sp]     // encoding: [0xe3,0x53,0x22,0xb8]
  // CHECK: ldsmina w0, w1, [x2]    // encoding: [0x41,0x50,0xa0,0xb8]
  // CHECK: ldsmina w2, w3, [sp]    // encoding: [0xe3,0x53,0xa2,0xb8]
  // CHECK: ldsminl w0, w1, [x2]    // encoding: [0x41,0x50,0x60,0xb8]
  // CHECK: ldsminl w2, w3, [sp]    // encoding: [0xe3,0x53,0x62,0xb8]
  // CHECK: ldsminal w0, w1, [x2]   // encoding: [0x41,0x50,0xe0,0xb8]
  // CHECK: ldsminal w2, w3, [sp]   // encoding: [0xe3,0x53,0xe2,0xb8]

  ldsminb w0, w1, [x2]
  ldsminb w2, w3, [sp]
  ldsminh w0, w1, [x2]
  ldsminh w2, w3, [sp]
  ldsminab w0, w1, [x2]
  ldsminab w2, w3, [sp]
  ldsminlb w0, w1, [x2]
  ldsminlb w2, w3, [sp]
  // CHECK: ldsminb w0, w1, [x2]      // encoding: [0x41,0x50,0x20,0x38]
  // CHECK: ldsminb w2, w3, [sp]      // encoding: [0xe3,0x53,0x22,0x38]
  // CHECK: ldsminh w0, w1, [x2]      // encoding: [0x41,0x50,0x20,0x78]
  // CHECK: ldsminh w2, w3, [sp]      // encoding: [0xe3,0x53,0x22,0x78]
  // CHECK: ldsminab w0, w1, [x2]     // encoding: [0x41,0x50,0xa0,0x38]
  // CHECK: ldsminab w2, w3, [sp]     // encoding: [0xe3,0x53,0xa2,0x38]
  // CHECK: ldsminlb w0, w1, [x2]     // encoding: [0x41,0x50,0x60,0x38]
  // CHECK: ldsminlb w2, w3, [sp]     // encoding: [0xe3,0x53,0x62,0x38]

  ldsminalb w0, w1, [x2]
  ldsminalb w2, w3, [sp]
  ldsminah w0, w1, [x2]
  ldsminah w2, w3, [sp]
  ldsminlh w0, w1, [x2]
  ldsminlh w2, w3, [sp]
  ldsminalh w0, w1, [x2]
  ldsminalh w2, w3, [sp]
  // CHECK: ldsminalb w0, w1, [x2]    // encoding: [0x41,0x50,0xe0,0x38]
  // CHECK: ldsminalb w2, w3, [sp]    // encoding: [0xe3,0x53,0xe2,0x38]
  // CHECK: ldsminah w0, w1, [x2]     // encoding: [0x41,0x50,0xa0,0x78]
  // CHECK: ldsminah w2, w3, [sp]     // encoding: [0xe3,0x53,0xa2,0x78]
  // CHECK: ldsminlh w0, w1, [x2]     // encoding: [0x41,0x50,0x60,0x78]
  // CHECK: ldsminlh w2, w3, [sp]     // encoding: [0xe3,0x53,0x62,0x78]
  // CHECK: ldsminalh w0, w1, [x2]    // encoding: [0x41,0x50,0xe0,0x78]
  // CHECK: ldsminalh w2, w3, [sp]    // encoding: [0xe3,0x53,0xe2,0x78]

  ldsmin x0, x1, [x2]
  ldsmin x2, x3, [sp]
  ldsmina x0, x1, [x2]
  ldsmina x2, x3, [sp]
  ldsminl x0, x1, [x2]
  ldsminl x2, x3, [sp]
  ldsminal x0, x1, [x2]
  ldsminal x2, x3, [sp]
  // CHECK: ldsmin x0, x1, [x2]     // encoding: [0x41,0x50,0x20,0xf8]
  // CHECK: ldsmin x2, x3, [sp]     // encoding: [0xe3,0x53,0x22,0xf8]
  // CHECK: ldsmina x0, x1, [x2]    // encoding: [0x41,0x50,0xa0,0xf8]
  // CHECK: ldsmina x2, x3, [sp]    // encoding: [0xe3,0x53,0xa2,0xf8]
  // CHECK: ldsminl x0, x1, [x2]    // encoding: [0x41,0x50,0x60,0xf8]
  // CHECK: ldsminl x2, x3, [sp]    // encoding: [0xe3,0x53,0x62,0xf8]
  // CHECK: ldsminal x0, x1, [x2]   // encoding: [0x41,0x50,0xe0,0xf8]
  // CHECK: ldsminal x2, x3, [sp]   // encoding: [0xe3,0x53,0xe2,0xf8]

  ldumax w0, w1, [x2]
  ldumax w2, w3, [sp]
  ldumaxa w0, w1, [x2]
  ldumaxa w2, w3, [sp]
  ldumaxl w0, w1, [x2]
  ldumaxl w2, w3, [sp]
  ldumaxal w0, w1, [x2]
  ldumaxal w2, w3, [sp]
  // CHECK: ldumax w0, w1, [x2]     // encoding: [0x41,0x60,0x20,0xb8]
  // CHECK: ldumax w2, w3, [sp]     // encoding: [0xe3,0x63,0x22,0xb8]
  // CHECK: ldumaxa w0, w1, [x2]    // encoding: [0x41,0x60,0xa0,0xb8]
  // CHECK: ldumaxa w2, w3, [sp]    // encoding: [0xe3,0x63,0xa2,0xb8]
  // CHECK: ldumaxl w0, w1, [x2]    // encoding: [0x41,0x60,0x60,0xb8]
  // CHECK: ldumaxl w2, w3, [sp]    // encoding: [0xe3,0x63,0x62,0xb8]
  // CHECK: ldumaxal w0, w1, [x2]   // encoding: [0x41,0x60,0xe0,0xb8]
  // CHECK: ldumaxal w2, w3, [sp]   // encoding: [0xe3,0x63,0xe2,0xb8]

  ldumaxb w0, w1, [x2]
  ldumaxb w2, w3, [sp]
  ldumaxh w0, w1, [x2]
  ldumaxh w2, w3, [sp]
  ldumaxab w0, w1, [x2]
  ldumaxab w2, w3, [sp]
  ldumaxlb w0, w1, [x2]
  ldumaxlb w2, w3, [sp]
  // CHECK: ldumaxb w0, w1, [x2]     // encoding: [0x41,0x60,0x20,0x38]
  // CHECK: ldumaxb w2, w3, [sp]     // encoding: [0xe3,0x63,0x22,0x38]
  // CHECK: ldumaxh w0, w1, [x2]     // encoding: [0x41,0x60,0x20,0x78]
  // CHECK: ldumaxh w2, w3, [sp]     // encoding: [0xe3,0x63,0x22,0x78]
  // CHECK: ldumaxab w0, w1, [x2]    // encoding: [0x41,0x60,0xa0,0x38]
  // CHECK: ldumaxab w2, w3, [sp]    // encoding: [0xe3,0x63,0xa2,0x38]
  // CHECK: ldumaxlb w0, w1, [x2]    // encoding: [0x41,0x60,0x60,0x38]
  // CHECK: ldumaxlb w2, w3, [sp]    // encoding: [0xe3,0x63,0x62,0x38]

  ldumaxalb w0, w1, [x2]
  ldumaxalb w2, w3, [sp]
  ldumaxah w0, w1, [x2]
  ldumaxah w2, w3, [sp]
  ldumaxlh w0, w1, [x2]
  ldumaxlh w2, w3, [sp]
  ldumaxalh w0, w1, [x2]
  ldumaxalh w2, w3, [sp]
  // CHECK: ldumaxalb w0, w1, [x2]    // encoding: [0x41,0x60,0xe0,0x38]
  // CHECK: ldumaxalb w2, w3, [sp]    // encoding: [0xe3,0x63,0xe2,0x38]
  // CHECK: ldumaxah w0, w1, [x2]     // encoding: [0x41,0x60,0xa0,0x78]
  // CHECK: ldumaxah w2, w3, [sp]     // encoding: [0xe3,0x63,0xa2,0x78]
  // CHECK: ldumaxlh w0, w1, [x2]     // encoding: [0x41,0x60,0x60,0x78]
  // CHECK: ldumaxlh w2, w3, [sp]     // encoding: [0xe3,0x63,0x62,0x78]
  // CHECK: ldumaxalh w0, w1, [x2]    // encoding: [0x41,0x60,0xe0,0x78]
  // CHECK: ldumaxalh w2, w3, [sp]    // encoding: [0xe3,0x63,0xe2,0x78]

  ldumax x0, x1, [x2]
  ldumax x2, x3, [sp]
  ldumaxa x0, x1, [x2]
  ldumaxa x2, x3, [sp]
  ldumaxl x0, x1, [x2]
  ldumaxl x2, x3, [sp]
  ldumaxal x0, x1, [x2]
  ldumaxal x2, x3, [sp]
  // CHECK: ldumax x0, x1, [x2]     // encoding: [0x41,0x60,0x20,0xf8]
  // CHECK: ldumax x2, x3, [sp]     // encoding: [0xe3,0x63,0x22,0xf8]
  // CHECK: ldumaxa x0, x1, [x2]    // encoding: [0x41,0x60,0xa0,0xf8]
  // CHECK: ldumaxa x2, x3, [sp]    // encoding: [0xe3,0x63,0xa2,0xf8]
  // CHECK: ldumaxl x0, x1, [x2]    // encoding: [0x41,0x60,0x60,0xf8]
  // CHECK: ldumaxl x2, x3, [sp]    // encoding: [0xe3,0x63,0x62,0xf8]
  // CHECK: ldumaxal x0, x1, [x2]   // encoding: [0x41,0x60,0xe0,0xf8]
  // CHECK: ldumaxal x2, x3, [sp]   // encoding: [0xe3,0x63,0xe2,0xf8]

  ldumin w0, w1, [x2]
  ldumin w2, w3, [sp]
  ldumina w0, w1, [x2]
  ldumina w2, w3, [sp]
  lduminl w0, w1, [x2]
  lduminl w2, w3, [sp]
  lduminal w0, w1, [x2]
  lduminal w2, w3, [sp]
  // CHECK: ldumin w0, w1, [x2]     // encoding: [0x41,0x70,0x20,0xb8]
  // CHECK: ldumin w2, w3, [sp]     // encoding: [0xe3,0x73,0x22,0xb8]
  // CHECK: ldumina w0, w1, [x2]    // encoding: [0x41,0x70,0xa0,0xb8]
  // CHECK: ldumina w2, w3, [sp]    // encoding: [0xe3,0x73,0xa2,0xb8]
  // CHECK: lduminl w0, w1, [x2]    // encoding: [0x41,0x70,0x60,0xb8]
  // CHECK: lduminl w2, w3, [sp]    // encoding: [0xe3,0x73,0x62,0xb8]
  // CHECK: lduminal w0, w1, [x2]   // encoding: [0x41,0x70,0xe0,0xb8]
  // CHECK: lduminal w2, w3, [sp]   // encoding: [0xe3,0x73,0xe2,0xb8]

  lduminb w0, w1, [x2]
  lduminb w2, w3, [sp]
  lduminh w0, w1, [x2]
  lduminh w2, w3, [sp]
  lduminab w0, w1, [x2]
  lduminab w2, w3, [sp]
  lduminlb w0, w1, [x2]
  lduminlb w2, w3, [sp]
  // CHECK: lduminb w0, w1, [x2]     // encoding: [0x41,0x70,0x20,0x38]
  // CHECK: lduminb w2, w3, [sp]     // encoding: [0xe3,0x73,0x22,0x38]
  // CHECK: lduminh w0, w1, [x2]     // encoding: [0x41,0x70,0x20,0x78]
  // CHECK: lduminh w2, w3, [sp]     // encoding: [0xe3,0x73,0x22,0x78]
  // CHECK: lduminab w0, w1, [x2]    // encoding: [0x41,0x70,0xa0,0x38]
  // CHECK: lduminab w2, w3, [sp]    // encoding: [0xe3,0x73,0xa2,0x38]
  // CHECK: lduminlb w0, w1, [x2]    // encoding: [0x41,0x70,0x60,0x38]
  // CHECK: lduminlb w2, w3, [sp]    // encoding: [0xe3,0x73,0x62,0x38]

  lduminalb w0, w1, [x2]
  lduminalb w2, w3, [sp]
  lduminah w0, w1, [x2]
  lduminah w2, w3, [sp]
  lduminlh w0, w1, [x2]
  lduminlh w2, w3, [sp]
  lduminalh w0, w1, [x2]
  lduminalh w2, w3, [sp]
  // CHECK: lduminalb w0, w1, [x2]    // encoding: [0x41,0x70,0xe0,0x38]
  // CHECK: lduminalb w2, w3, [sp]    // encoding: [0xe3,0x73,0xe2,0x38]
  // CHECK: lduminah w0, w1, [x2]     // encoding: [0x41,0x70,0xa0,0x78]
  // CHECK: lduminah w2, w3, [sp]     // encoding: [0xe3,0x73,0xa2,0x78]
  // CHECK: lduminlh w0, w1, [x2]     // encoding: [0x41,0x70,0x60,0x78]
  // CHECK: lduminlh w2, w3, [sp]     // encoding: [0xe3,0x73,0x62,0x78]
  // CHECK: lduminalh w0, w1, [x2]    // encoding: [0x41,0x70,0xe0,0x78]
  // CHECK: lduminalh w2, w3, [sp]    // encoding: [0xe3,0x73,0xe2,0x78]

  ldumin x0, x1, [x2]
  ldumin x2, x3, [sp]
  ldumina x0, x1, [x2]
  ldumina x2, x3, [sp]
  lduminl x0, x1, [x2]
  lduminl x2, x3, [sp]
  lduminal x0, x1, [x2]
  lduminal x2, x3, [sp]
  // CHECK: ldumin x0, x1, [x2]     // encoding: [0x41,0x70,0x20,0xf8]
  // CHECK: ldumin x2, x3, [sp]     // encoding: [0xe3,0x73,0x22,0xf8]
  // CHECK: ldumina x0, x1, [x2]    // encoding: [0x41,0x70,0xa0,0xf8]
  // CHECK: ldumina x2, x3, [sp]    // encoding: [0xe3,0x73,0xa2,0xf8]
  // CHECK: lduminl x0, x1, [x2]    // encoding: [0x41,0x70,0x60,0xf8]
  // CHECK: lduminl x2, x3, [sp]    // encoding: [0xe3,0x73,0x62,0xf8]
  // CHECK: lduminal x0, x1, [x2]   // encoding: [0x41,0x70,0xe0,0xf8]
  // CHECK: lduminal x2, x3, [sp]   // encoding: [0xe3,0x73,0xe2,0xf8]

  stadd w0, [x2]
  stadd w2, [sp]
  staddl w0, [x2]
  staddl w2, [sp]
  staddb w0, [x2]
  staddb w2, [sp]
  staddh w0, [x2]
  staddh w2, [sp]
  // CHECK: stadd w0, [x2]      // encoding: [0x5f,0x00,0x20,0xb8]
  // CHECK: stadd w2, [sp]      // encoding: [0xff,0x03,0x22,0xb8]
  // CHECK: staddl w0, [x2]     // encoding: [0x5f,0x00,0x60,0xb8]
  // CHECK: staddl w2, [sp]     // encoding: [0xff,0x03,0x62,0xb8]
  // CHECK: staddb w0, [x2]     // encoding: [0x5f,0x00,0x20,0x38]
  // CHECK: staddb w2, [sp]     // encoding: [0xff,0x03,0x22,0x38]
  // CHECK: staddh w0, [x2]     // encoding: [0x5f,0x00,0x20,0x78]
  // CHECK: staddh w2, [sp]     // encoding: [0xff,0x03,0x22,0x78]

  staddlb w0, [x2]
  staddlb w2, [sp]
  staddlh w0, [x2]
  staddlh w2, [sp]
  stadd x0, [x2]
  stadd x2, [sp]
  staddl x0, [x2]
  staddl x2, [sp]
  // CHECK: staddlb w0, [x2]    // encoding: [0x5f,0x00,0x60,0x38]
  // CHECK: staddlb w2, [sp]    // encoding: [0xff,0x03,0x62,0x38]
  // CHECK: staddlh w0, [x2]    // encoding: [0x5f,0x00,0x60,0x78]
  // CHECK: staddlh w2, [sp]    // encoding: [0xff,0x03,0x62,0x78]
  // CHECK: stadd x0, [x2]      // encoding: [0x5f,0x00,0x20,0xf8]
  // CHECK: stadd x2, [sp]      // encoding: [0xff,0x03,0x22,0xf8]
  // CHECK: staddl x0, [x2]     // encoding: [0x5f,0x00,0x60,0xf8]
  // CHECK: staddl x2, [sp]     // encoding: [0xff,0x03,0x62,0xf8]

  stclr w0, [x2]
  stclr w2, [sp]
  stclrl w0, [x2]
  stclrl w2, [sp]
  stclrb w0, [x2]
  stclrb w2, [sp]
  stclrh w0, [x2]
  stclrh w2, [sp]
  // CHECK: stclr w0, [x2]      // encoding: [0x5f,0x10,0x20,0xb8]
  // CHECK: stclr w2, [sp]      // encoding: [0xff,0x13,0x22,0xb8]
  // CHECK: stclrl w0, [x2]     // encoding: [0x5f,0x10,0x60,0xb8]
  // CHECK: stclrl w2, [sp]     // encoding: [0xff,0x13,0x62,0xb8]
  // CHECK: stclrb w0, [x2]     // encoding: [0x5f,0x10,0x20,0x38]
  // CHECK: stclrb w2, [sp]     // encoding: [0xff,0x13,0x22,0x38]
  // CHECK: stclrh w0, [x2]     // encoding: [0x5f,0x10,0x20,0x78]
  // CHECK: stclrh w2, [sp]     // encoding: [0xff,0x13,0x22,0x78]

  stclrlb w0, [x2]
  stclrlb w2, [sp]
  stclrlh w0, [x2]
  stclrlh w2, [sp]
  stclr x0, [x2]
  stclr x2, [sp]
  stclrl x0, [x2]
  stclrl x2, [sp]
  // CHECK: stclrlb w0, [x2]    // encoding: [0x5f,0x10,0x60,0x38]
  // CHECK: stclrlb w2, [sp]    // encoding: [0xff,0x13,0x62,0x38]
  // CHECK: stclrlh w0, [x2]    // encoding: [0x5f,0x10,0x60,0x78]
  // CHECK: stclrlh w2, [sp]    // encoding: [0xff,0x13,0x62,0x78]
  // CHECK: stclr x0, [x2]      // encoding: [0x5f,0x10,0x20,0xf8]
  // CHECK: stclr x2, [sp]      // encoding: [0xff,0x13,0x22,0xf8]
  // CHECK: stclrl x0, [x2]     // encoding: [0x5f,0x10,0x60,0xf8]
  // CHECK: stclrl x2, [sp]     // encoding: [0xff,0x13,0x62,0xf8]

  steor w0, [x2]
  steor w2, [sp]
  steorl w0, [x2]
  steorl w2, [sp]
  steorb w0, [x2]
  steorb w2, [sp]
  steorh w0, [x2]
  steorh w2, [sp]
  // CHECK: steor w0, [x2]      // encoding: [0x5f,0x20,0x20,0xb8]
  // CHECK: steor w2, [sp]      // encoding: [0xff,0x23,0x22,0xb8]
  // CHECK: steorl w0, [x2]     // encoding: [0x5f,0x20,0x60,0xb8]
  // CHECK: steorl w2, [sp]     // encoding: [0xff,0x23,0x62,0xb8]
  // CHECK: steorb w0, [x2]     // encoding: [0x5f,0x20,0x20,0x38]
  // CHECK: steorb w2, [sp]     // encoding: [0xff,0x23,0x22,0x38]
  // CHECK: steorh w0, [x2]     // encoding: [0x5f,0x20,0x20,0x78]
  // CHECK: steorh w2, [sp]     // encoding: [0xff,0x23,0x22,0x78]

  steorlb w0, [x2]
  steorlb w2, [sp]
  steorlh w0, [x2]
  steorlh w2, [sp]
  steor x0, [x2]
  steor x2, [sp]
  steorl x0, [x2]
  steorl x2, [sp]
  // CHECK: steorlb w0, [x2]    // encoding: [0x5f,0x20,0x60,0x38]
  // CHECK: steorlb w2, [sp]    // encoding: [0xff,0x23,0x62,0x38]
  // CHECK: steorlh w0, [x2]    // encoding: [0x5f,0x20,0x60,0x78]
  // CHECK: steorlh w2, [sp]    // encoding: [0xff,0x23,0x62,0x78]
  // CHECK: steor x0, [x2]      // encoding: [0x5f,0x20,0x20,0xf8]
  // CHECK: steor x2, [sp]      // encoding: [0xff,0x23,0x22,0xf8]
  // CHECK: steorl x0, [x2]     // encoding: [0x5f,0x20,0x60,0xf8]
  // CHECK: steorl x2, [sp]     // encoding: [0xff,0x23,0x62,0xf8]

  stset w0, [x2]
  stset w2, [sp]
  stsetl w0, [x2]
  stsetl w2, [sp]
  stsetb w0, [x2]
  stsetb w2, [sp]
  stseth w0, [x2]
  stseth w2, [sp]
  // CHECK: stset w0, [x2]      // encoding: [0x5f,0x30,0x20,0xb8]
  // CHECK: stset w2, [sp]      // encoding: [0xff,0x33,0x22,0xb8]
  // CHECK: stsetl w0, [x2]     // encoding: [0x5f,0x30,0x60,0xb8]
  // CHECK: stsetl w2, [sp]     // encoding: [0xff,0x33,0x62,0xb8]
  // CHECK: stsetb w0, [x2]     // encoding: [0x5f,0x30,0x20,0x38]
  // CHECK: stsetb w2, [sp]     // encoding: [0xff,0x33,0x22,0x38]
  // CHECK: stseth w0, [x2]     // encoding: [0x5f,0x30,0x20,0x78]
  // CHECK: stseth w2, [sp]     // encoding: [0xff,0x33,0x22,0x78]

  stsetlb w0, [x2]
  stsetlb w2, [sp]
  stsetlh w0, [x2]
  stsetlh w2, [sp]
  stset x0, [x2]
  stset x2, [sp]
  stsetl x0, [x2]
  stsetl x2, [sp]
  // CHECK: stsetlb w0, [x2]    // encoding: [0x5f,0x30,0x60,0x38]
  // CHECK: stsetlb w2, [sp]    // encoding: [0xff,0x33,0x62,0x38]
  // CHECK: stsetlh w0, [x2]    // encoding: [0x5f,0x30,0x60,0x78]
  // CHECK: stsetlh w2, [sp]    // encoding: [0xff,0x33,0x62,0x78]
  // CHECK: stset x0, [x2]      // encoding: [0x5f,0x30,0x20,0xf8]
  // CHECK: stset x2, [sp]      // encoding: [0xff,0x33,0x22,0xf8]
  // CHECK: stsetl x0, [x2]     // encoding: [0x5f,0x30,0x60,0xf8]
  // CHECK: stsetl x2, [sp]     // encoding: [0xff,0x33,0x62,0xf8]

  stsmax w0, [x2]
  stsmax w2, [sp]
  stsmaxl w0, [x2]
  stsmaxl w2, [sp]
  stsmaxb w0, [x2]
  stsmaxb w2, [sp]
  stsmaxh w0, [x2]
  stsmaxh w2, [sp]
  // CHECK: stsmax w0, [x2]     // encoding: [0x5f,0x40,0x20,0xb8]
  // CHECK: stsmax w2, [sp]     // encoding: [0xff,0x43,0x22,0xb8]
  // CHECK: stsmaxl w0, [x2]    // encoding: [0x5f,0x40,0x60,0xb8]
  // CHECK: stsmaxl w2, [sp]    // encoding: [0xff,0x43,0x62,0xb8]
  // CHECK: stsmaxb w0, [x2]    // encoding: [0x5f,0x40,0x20,0x38]
  // CHECK: stsmaxb w2, [sp]    // encoding: [0xff,0x43,0x22,0x38]
  // CHECK: stsmaxh w0, [x2]    // encoding: [0x5f,0x40,0x20,0x78]
  // CHECK: stsmaxh w2, [sp]    // encoding: [0xff,0x43,0x22,0x78]

  stsmaxlb w0, [x2]
  stsmaxlb w2, [sp]
  stsmaxlh w0, [x2]
  stsmaxlh w2, [sp]
  stsmax x0, [x2]
  stsmax x2, [sp]
  stsmaxl x0, [x2]
  stsmaxl x2, [sp]
  // CHECK: stsmaxlb w0, [x2]   // encoding: [0x5f,0x40,0x60,0x38]
  // CHECK: stsmaxlb w2, [sp]   // encoding: [0xff,0x43,0x62,0x38]
  // CHECK: stsmaxlh w0, [x2]   // encoding: [0x5f,0x40,0x60,0x78]
  // CHECK: stsmaxlh w2, [sp]   // encoding: [0xff,0x43,0x62,0x78]
  // CHECK: stsmax x0, [x2]     // encoding: [0x5f,0x40,0x20,0xf8]
  // CHECK: stsmax x2, [sp]     // encoding: [0xff,0x43,0x22,0xf8]
  // CHECK: stsmaxl x0, [x2]    // encoding: [0x5f,0x40,0x60,0xf8]
  // CHECK: stsmaxl x2, [sp]    // encoding: [0xff,0x43,0x62,0xf8]

  stsmin w0, [x2]
  stsmin w2, [sp]
  stsminl w0, [x2]
  stsminl w2, [sp]
  stsminb w0, [x2]
  stsminb w2, [sp]
  stsminh w0, [x2]
  stsminh w2, [sp]
  // CHECK: stsmin w0, [x2]     // encoding: [0x5f,0x50,0x20,0xb8]
  // CHECK: stsmin w2, [sp]     // encoding: [0xff,0x53,0x22,0xb8]
  // CHECK: stsminl w0, [x2]    // encoding: [0x5f,0x50,0x60,0xb8]
  // CHECK: stsminl w2, [sp]    // encoding: [0xff,0x53,0x62,0xb8]
  // CHECK: stsminb w0, [x2]    // encoding: [0x5f,0x50,0x20,0x38]
  // CHECK: stsminb w2, [sp]    // encoding: [0xff,0x53,0x22,0x38]
  // CHECK: stsminh w0, [x2]    // encoding: [0x5f,0x50,0x20,0x78]
  // CHECK: stsminh w2, [sp]    // encoding: [0xff,0x53,0x22,0x78]

  stsminlb w0, [x2]
  stsminlb w2, [sp]
  stsminlh w0, [x2]
  stsminlh w2, [sp]
  stsmin x0, [x2]
  stsmin x2, [sp]
  stsminl x0, [x2]
  stsminl x2, [sp]
  // CHECK: stsminlb w0, [x2]   // encoding: [0x5f,0x50,0x60,0x38]
  // CHECK: stsminlb w2, [sp]   // encoding: [0xff,0x53,0x62,0x38]
  // CHECK: stsminlh w0, [x2]   // encoding: [0x5f,0x50,0x60,0x78]
  // CHECK: stsminlh w2, [sp]   // encoding: [0xff,0x53,0x62,0x78]
  // CHECK: stsmin x0, [x2]     // encoding: [0x5f,0x50,0x20,0xf8]
  // CHECK: stsmin x2, [sp]     // encoding: [0xff,0x53,0x22,0xf8]
  // CHECK: stsminl x0, [x2]    // encoding: [0x5f,0x50,0x60,0xf8]
  // CHECK: stsminl x2, [sp]    // encoding: [0xff,0x53,0x62,0xf8]

  stumax w0, [x2]
  stumax w2, [sp]
  stumaxl w0, [x2]
  stumaxl w2, [sp]
  stumaxb w0, [x2]
  stumaxb w2, [sp]
  stumaxh w0, [x2]
  stumaxh w2, [sp]
  // CHECK: stumax w0, [x2]     // encoding: [0x5f,0x60,0x20,0xb8]
  // CHECK: stumax w2, [sp]     // encoding: [0xff,0x63,0x22,0xb8]
  // CHECK: stumaxl w0, [x2]    // encoding: [0x5f,0x60,0x60,0xb8]
  // CHECK: stumaxl w2, [sp]    // encoding: [0xff,0x63,0x62,0xb8]
  // CHECK: stumaxb w0, [x2]    // encoding: [0x5f,0x60,0x20,0x38]
  // CHECK: stumaxb w2, [sp]    // encoding: [0xff,0x63,0x22,0x38]
  // CHECK: stumaxh w0, [x2]    // encoding: [0x5f,0x60,0x20,0x78]
  // CHECK: stumaxh w2, [sp]    // encoding: [0xff,0x63,0x22,0x78]

  stumaxlb w0, [x2]
  stumaxlb w2, [sp]
  stumaxlh w0, [x2]
  stumaxlh w2, [sp]
  stumax x0, [x2]
  stumax x2, [sp]
  stumaxl x0, [x2]
  stumaxl x2, [sp]
  // CHECK: stumaxlb w0, [x2]   // encoding: [0x5f,0x60,0x60,0x38]
  // CHECK: stumaxlb w2, [sp]   // encoding: [0xff,0x63,0x62,0x38]
  // CHECK: stumaxlh w0, [x2]   // encoding: [0x5f,0x60,0x60,0x78]
  // CHECK: stumaxlh w2, [sp]   // encoding: [0xff,0x63,0x62,0x78]
  // CHECK: stumax x0, [x2]     // encoding: [0x5f,0x60,0x20,0xf8]
  // CHECK: stumax x2, [sp]     // encoding: [0xff,0x63,0x22,0xf8]
  // CHECK: stumaxl x0, [x2]    // encoding: [0x5f,0x60,0x60,0xf8]
  // CHECK: stumaxl x2, [sp]    // encoding: [0xff,0x63,0x62,0xf8]

  stumin w0, [x2]
  stumin w2, [sp]
  stuminl w0, [x2]
  stuminl w2, [sp]
  stuminb w0, [x2]
  stuminb w2, [sp]
  stuminh w0, [x2]
  stuminh w2, [sp]
  // CHECK: stumin w0, [x2]     // encoding: [0x5f,0x70,0x20,0xb8]
  // CHECK: stumin w2, [sp]     // encoding: [0xff,0x73,0x22,0xb8]
  // CHECK: stuminl w0, [x2]    // encoding: [0x5f,0x70,0x60,0xb8]
  // CHECK: stuminl w2, [sp]    // encoding: [0xff,0x73,0x62,0xb8]
  // CHECK: stuminb w0, [x2]    // encoding: [0x5f,0x70,0x20,0x38]
  // CHECK: stuminb w2, [sp]    // encoding: [0xff,0x73,0x22,0x38]
  // CHECK: stuminh w0, [x2]    // encoding: [0x5f,0x70,0x20,0x78]
  // CHECK: stuminh w2, [sp]    // encoding: [0xff,0x73,0x22,0x78]

  cas b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cas b0, b1, [x2]
  // CHECK-ERROR:       ^

  cas b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cas b2, b3, [sp]
  // CHECK-ERROR:       ^

  cas h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cas h0, h1, [x2]
  // CHECK-ERROR:       ^

  cas h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cas h2, h3, [sp]
  // CHECK-ERROR:       ^

  casa b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casa b0, b1, [x2]
  // CHECK-ERROR:        ^

  casa b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casa b2, b3, [sp]
  // CHECK-ERROR:        ^

  casa h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casa h0, h1, [x2]
  // CHECK-ERROR:        ^

  casa h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casa h2, h3, [sp]
  // CHECK-ERROR:        ^

  casb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casb b0, b1, [x2]
  // CHECK-ERROR:        ^

  casb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casb b2, b3, [sp]
  // CHECK-ERROR:        ^

  casb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casb h0, h1, [x2]
  // CHECK-ERROR:        ^

  casb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casb h2, h3, [sp]
  // CHECK-ERROR:        ^

  cash b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cash b0, b1, [x2]
  // CHECK-ERROR:        ^

  cash b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cash b2, b3, [sp]
  // CHECK-ERROR:        ^

  cash h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cash h0, h1, [x2]
  // CHECK-ERROR:        ^

  cash h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   cash h2, h3, [sp]
  // CHECK-ERROR:        ^

  casah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casah b0, b1, [x2]
  // CHECK-ERROR:         ^

  casah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casah b2, b3, [sp]
  // CHECK-ERROR:         ^

  casah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casah h0, h1, [x2]
  // CHECK-ERROR:         ^

  casah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casah h2, h3, [sp]
  // CHECK-ERROR:         ^

  casalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh b0, b1, [x2]
  // CHECK-ERROR:          ^

  casalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh b2, b3, [sp]
  // CHECK-ERROR:          ^

  casalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh h0, h1, [x2]
  // CHECK-ERROR:          ^

  casalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh h2, h3, [sp]
  // CHECK-ERROR:          ^


  casl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casl b0, b1, [x2]
  // CHECK-ERROR:        ^

  casl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casl b2, b3, [sp]
  // CHECK-ERROR:        ^

  casl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casl h0, h1, [x2]
  // CHECK-ERROR:        ^

  casl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casl h2, h3, [sp]
  // CHECK-ERROR:        ^

  caslb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   caslb b0, b1, [x2]
  // CHECK-ERROR:         ^

  caslb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   caslb b2, b3, [sp]
  // CHECK-ERROR:         ^

  caslb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   caslb h0, h1, [x2]
  // CHECK-ERROR:         ^

  caslb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   caslb h2, h3, [sp]
  // CHECK-ERROR:         ^


  casalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalb b0, b1, [x2]
  // CHECK-ERROR:          ^

  casalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalb b2, b3, [sp]
  // CHECK-ERROR:          ^

  casalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalb h0, h1, [x2]
  // CHECK-ERROR:          ^

  casalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalb h2, h3, [sp]
  // CHECK-ERROR:          ^

  casalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh b0, b1, [x2]
  // CHECK-ERROR:          ^

  casalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh b2, b3, [sp]
  // CHECK-ERROR:          ^

  casalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh h0, h1, [x2]
  // CHECK-ERROR:          ^

  casalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh h2, h3, [sp]
  // CHECK-ERROR:          ^

  cas v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:   ^

  casa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:   ^

  casl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:   ^

  casal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  casb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:        ^

  casab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  caslb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   caslb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  casalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  casah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  caslh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   caslh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  casalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   casalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  casp b0, b1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   casp b0, b1, [x2]
  // CHECK-ERROR:         ^

  casp b2, b3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   casp b2, b3, [sp]
  // CHECK-ERROR:         ^

  casp h0, h1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   casp h0, h1, [x2]
  // CHECK-ERROR:         ^

  casp h2, h3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   casp h2, h3, [sp]
  // CHECK-ERROR:         ^

  caspa b0, b1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspa b0, b1, [x2]
  // CHECK-ERROR:         ^

  caspa b2, b3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspa b2, b3, [sp]
  // CHECK-ERROR:         ^

  caspa h0, h1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspa h0, h1, [x2]
  // CHECK-ERROR:         ^

  caspa h2, h3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspa h2, h3, [sp]
  // CHECK-ERROR:         ^

  caspl b0, b1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspl b0, b1, [x2]
  // CHECK-ERROR:         ^

  caspl b2, b3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspl b2, b3, [sp]
  // CHECK-ERROR:         ^

  caspl h0, h1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspl h0, h1, [x2]
  // CHECK-ERROR:         ^

  caspl h2, h3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspl h2, h3, [sp]
  // CHECK-ERROR:         ^

  caspal b0, b1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspal b0, b1, [x2]
  // CHECK-ERROR:         ^

  caspal b2, b3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspal b2, b3, [sp]
  // CHECK-ERROR:         ^

  caspal h0, h1, [x2]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspal h0, h1, [x2]
  // CHECK-ERROR:         ^

  caspal h2, h3, [sp]
  // CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
  // CHECK-ERROR:   caspal h2, h3, [sp]
  // CHECK-ERROR:         ^

  swp b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swp b0, b1, [x2]
  // CHECK-ERROR:       ^

  swp b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swp b2, b3, [sp]
  // CHECK-ERROR:       ^

  swpa b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpa b0, b1, [x2]
  // CHECK-ERROR:        ^

  swpa b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpa b2, b3, [sp]
  // CHECK-ERROR:        ^

  swpah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpah b0, b1, [x2]
  // CHECK-ERROR:         ^

  swpah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpah b2, b3, [sp]
  // CHECK-ERROR:         ^

  swpl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpl b0, b1, [x2]
  // CHECK-ERROR:        ^

  swpl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpl b2, b3, [sp]
  // CHECK-ERROR:        ^

  swpal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpal b0, b1, [x2]
  // CHECK-ERROR:         ^

  swpal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpal b2, b3, [sp]
  // CHECK-ERROR:         ^

  swpalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalb b0, b1, [x2]
  // CHECK-ERROR:          ^

  swpalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalb b2, b3, [sp]
  // CHECK-ERROR:          ^

  swpalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalh b0, b1, [x2]
  // CHECK-ERROR:          ^

  swpalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalh b2, b3, [sp]
  // CHECK-ERROR:          ^

  swpb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpb b0, b1, [x2]
  // CHECK-ERROR:        ^

  swpb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpb b2, b3, [sp]
  // CHECK-ERROR:        ^

  swpab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpab b0, b1, [x2]
  // CHECK-ERROR:         ^

  swpab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpab b2, b3, [sp]
  // CHECK-ERROR:         ^

  swpal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpal b0, b1, [x2]
  // CHECK-ERROR:         ^

  swpal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpal b2, b3, [sp]
  // CHECK-ERROR:         ^

  swpah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpah b0, b1, [x2]
  // CHECK-ERROR:         ^

  swpah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpah b2, b3, [sp]
  // CHECK-ERROR:         ^

  swpalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalh b0, b1, [x2]
  // CHECK-ERROR:          ^

  swpalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalh b2, b3, [sp]
  // CHECK-ERROR:          ^

  swpl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpl b0, b1, [x2]
  // CHECK-ERROR:        ^

  swpl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpl b2, b3, [sp]
  // CHECK-ERROR:        ^

  swplb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swplb b0, b1, [x2]
  // CHECK-ERROR:         ^

  swplb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swplb b2, b3, [sp]
  // CHECK-ERROR:         ^

  swpalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalb b0, b1, [x2]
  // CHECK-ERROR:          ^

  swpalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalb b2, b3, [sp]
  // CHECK-ERROR:          ^

  swph b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swph b0, b1, [x2]
  // CHECK-ERROR:        ^

  swph b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swph b2, b3, [sp]
  // CHECK-ERROR:        ^

  swp v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swp v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:       ^

  swpa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:        ^

  swpah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  swpl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:        ^

  swpal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  swpalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  swpalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  swpb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:        ^

  swpab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  swpal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  swpah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  swpalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  swpl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:        ^

  swplb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swplb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  swpalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swpalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  swph v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   swph v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:        ^

  ldadd b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadd b0, b1, [x2]
  // CHECK-ERROR:         ^

  ldadd b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadd b2, b3, [sp]
  // CHECK-ERROR:         ^

  ldadd h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadd h0, h1, [x2]
  // CHECK-ERROR:         ^

  ldadd h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadd h2, h3, [sp]
  // CHECK-ERROR:         ^

  ldadd v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadd v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  ldadda b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadda b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldadda b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadda b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldadda h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadda h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldadda h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadda h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldadda v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldadda v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldaddl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddl b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldaddl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddl b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldaddl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddl h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldaddl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddl h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldaddl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldaddal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddal b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldaddal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddal b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldaddal h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddal h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldaddal h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddal h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldaddal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldaddb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddb b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldaddb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddb b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldaddb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddb h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldaddb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddb h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldaddb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldaddh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddh b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldaddh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddh b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldaddh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddh h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldaddh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddh h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldaddh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldaddab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddab b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldaddab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddab b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldaddab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddab h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldaddab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddab h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldaddab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldaddlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldaddlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldaddlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldaddlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldaddlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldaddalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldaddalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldaddalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldaddalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldaddalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldaddah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddah b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldaddah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddah b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldaddah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddah h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldaddah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddah h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldaddah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldaddlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldaddlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldaddlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldaddlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldaddlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldaddalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldaddalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldaddalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldaddalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldaddalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldaddalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldclr b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclr b0, b1, [x2]
  // CHECK-ERROR:         ^

  ldclr b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclr b2, b3, [sp]
  // CHECK-ERROR:         ^

  ldclr h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclr h0, h1, [x2]
  // CHECK-ERROR:         ^

  ldclr h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclr h2, h3, [sp]
  // CHECK-ERROR:         ^

  ldclr v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclr v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  ldclra b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldclra b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldclra h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldclra h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldclra v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldclra b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldclra b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldclra h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldclra h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldclra v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclra v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldclrl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrl b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldclrl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrl b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldclrl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrl h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldclrl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrl h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldclrl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldclral b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclral b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldclral b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclral b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldclral h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclral h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldclral h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclral h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldclral v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclral v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldclrb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrb b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldclrb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrb b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldclrb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrb h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldclrb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrb h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldclrb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldclrh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrh b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldclrh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrh b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldclrh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrh h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldclrh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrh h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldclrh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldclrab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrab b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldclrab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrab b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldclrab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrab h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldclrab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrab h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldclrab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldclrlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldclrlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldclrlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldclrlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldclrlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldclralb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldclralb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldclralb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldclralb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldclralb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldclrah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrah b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldclrah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrah b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldclrah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrah h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldclrah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrah h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldclrah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldclrlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldclrlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldclrlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldclrlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldclrlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclrlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldclralh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldclralh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldclralh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldclralh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldclralh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldclralh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldeor b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeor b0, b1, [x2]
  // CHECK-ERROR:         ^

  ldeor b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeor b2, b3, [sp]
  // CHECK-ERROR:         ^

  ldeor h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeor h0, h1, [x2]
  // CHECK-ERROR:         ^

  ldeor h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeor h2, h3, [sp]
  // CHECK-ERROR:         ^

  ldeor v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeor v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  ldeora b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeora b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldeora b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeora b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldeora h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeora h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldeora h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeora h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldeora v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeora v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldeorl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorl b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldeorl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorl b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldeorl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorl h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldeorl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorl h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldeorl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldeoral b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoral b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldeoral b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoral b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldeoral h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoral h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldeoral h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoral h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldeoral v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoral v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldeorb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorb b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldeorb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorb b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldeorb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorb h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldeorb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorb h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldeorb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldeorh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorh b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldeorh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorh b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldeorh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorh h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldeorh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorh h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldeorh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldeorab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorab b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldeorab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorab b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldeorab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorab h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldeorab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorab h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldeorab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldeorlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldeorlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldeorlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldeorlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldeorlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldeoralb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldeoralb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldeoralb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldeoralb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldeoralb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldeorah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorah b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldeorah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorah b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldeorah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorah h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldeorah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorah h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldeorah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldeorlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldeorlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldeorlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldeorlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldeorlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeorlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldeoralh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldeoralh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldeoralh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldeoralh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldeoralh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldeoralh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldset b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldset b0, b1, [x2]
  // CHECK-ERROR:         ^

  ldset b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldset b2, b3, [sp]
  // CHECK-ERROR:         ^

  ldset h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldset h0, h1, [x2]
  // CHECK-ERROR:         ^

  ldset h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldset h2, h3, [sp]
  // CHECK-ERROR:         ^

  ldset v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldset v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:         ^

  ldseta b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseta b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldseta b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseta b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldseta h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseta h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldseta h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseta h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldseta v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseta v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldsetl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetl b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldsetl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetl b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldsetl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetl h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldsetl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetl h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldsetl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldsetal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetal b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsetal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetal b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsetal h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetal h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsetal h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetal h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsetal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsetb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetb b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldsetb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetb b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldsetb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetb h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldsetb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetb h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldsetb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldseth b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseth b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldseth b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseth b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldseth h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseth h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldseth h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseth h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldseth v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldseth v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldsetab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetab b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsetab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetab b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsetab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetab h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsetab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetab h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsetab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsetlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsetlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsetlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsetlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsetlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsetalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsetalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsetalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsetalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsetalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsetah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetah b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsetah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetah b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsetah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetah h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsetah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetah h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsetah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsetlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsetlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsetlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsetlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsetlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsetalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsetalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsetalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsetalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsetalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsetalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsmax b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmax b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldsmax b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmax b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldsmax h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmax h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldsmax h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmax h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldsmax v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmax v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldsmaxa b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxa b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxa b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxa b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxa h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxa h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxa h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxa h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsmaxl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxl b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxl b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxl h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxl h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsmaxal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxal b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxal b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxal h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxal h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxal h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxal h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsmaxb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsmaxh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsmaxh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsmaxh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsmaxab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxab b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxab b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxab h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxab h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsmaxlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsmaxalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalb b0, b1, [x2]
  // CHECK-ERROR:             ^

  ldsmaxalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalb b2, b3, [sp]
  // CHECK-ERROR:             ^

  ldsmaxalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalb h0, h1, [x2]
  // CHECK-ERROR:             ^

  ldsmaxalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalb h2, h3, [sp]
  // CHECK-ERROR:             ^

  ldsmaxalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  ldsmaxah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxah b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxah b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxah h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxah h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsmaxlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsmaxlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsmaxlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsmaxalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalh b0, b1, [x2]
  // CHECK-ERROR:             ^

  ldsmaxalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalh b2, b3, [sp]
  // CHECK-ERROR:             ^

  ldsmaxalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalh h0, h1, [x2]
  // CHECK-ERROR:             ^

  ldsmaxalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalh h2, h3, [sp]
  // CHECK-ERROR:             ^

  ldsmaxalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmaxalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  ldsmin b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmin b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldsmin b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmin b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldsmin h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmin h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldsmin h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmin h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldsmin v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmin v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldsmina b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmina b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsmina b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmina b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsmina h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmina h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsmina h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmina h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsmina v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsmina v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsminl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminl b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsminl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminl b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsminl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminl h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsminl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminl h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsminl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsminal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminal b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsminal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminal b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsminal h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminal h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsminal h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminal h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsminal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsminb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsminb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsminb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsminb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsminb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsminh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldsminh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldsminh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldsminh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldsminh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldsminab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminab b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsminab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminab b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsminab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminab h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsminab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminab h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsminab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsminlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsminlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsminlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsminlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsminlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsminalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalb b0, b1, [x2]
  // CHECK-ERROR:             ^

  ldsminalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalb b2, b3, [sp]
  // CHECK-ERROR:             ^

  ldsminalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalb h0, h1, [x2]
  // CHECK-ERROR:             ^

  ldsminalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalb h2, h3, [sp]
  // CHECK-ERROR:             ^

  ldsminalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  ldsminah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminah b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsminah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminah b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsminah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminah h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsminah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminah h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsminah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsminlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldsminlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldsminlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldsminlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldsminlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldsminalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalh b0, b1, [x2]
  // CHECK-ERROR:             ^

  ldsminalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalh b2, b3, [sp]
  // CHECK-ERROR:             ^

  ldsminalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalh h0, h1, [x2]
  // CHECK-ERROR:             ^

  ldsminalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalh h2, h3, [sp]
  // CHECK-ERROR:             ^

  ldsminalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldsminalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  ldumax b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumax b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldumax b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumax b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldumax h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumax h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldumax h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumax h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldumax v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumax v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldumaxa b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxa b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldumaxa b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxa b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldumaxa h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxa h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldumaxa h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxa h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldumaxa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxa v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldumaxl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxl b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldumaxl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxl b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldumaxl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxl h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldumaxl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxl h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldumaxl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldumaxal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxal b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldumaxal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxal b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldumaxal h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxal h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldumaxal h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxal h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldumaxal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldumaxb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxb b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldumaxb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxb b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldumaxb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxb h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldumaxb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxb h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldumaxb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldumaxh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxh b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldumaxh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxh b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldumaxh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxh h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldumaxh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxh h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldumaxh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  ldumaxab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxab b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldumaxab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxab b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldumaxab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxab h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldumaxab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxab h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldumaxab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldumaxlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlb b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldumaxlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlb b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldumaxlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlb h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldumaxlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlb h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldumaxlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldumaxalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalb b0, b1, [x2]
  // CHECK-ERROR:             ^

  ldumaxalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalb b2, b3, [sp]
  // CHECK-ERROR:             ^

  ldumaxalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalb h0, h1, [x2]
  // CHECK-ERROR:             ^

  ldumaxalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalb h2, h3, [sp]
  // CHECK-ERROR:             ^

  ldumaxalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  ldumaxah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxah b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldumaxah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxah b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldumaxah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxah h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldumaxah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxah h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldumaxah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldumaxlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlh b0, b1, [x2]
  // CHECK-ERROR:            ^

  ldumaxlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlh b2, b3, [sp]
  // CHECK-ERROR:            ^

  ldumaxlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlh h0, h1, [x2]
  // CHECK-ERROR:            ^

  ldumaxlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlh h2, h3, [sp]
  // CHECK-ERROR:            ^

  ldumaxlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  ldumaxalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalh b0, b1, [x2]
  // CHECK-ERROR:             ^

  ldumaxalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalh b2, b3, [sp]
  // CHECK-ERROR:             ^

  ldumaxalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalh h0, h1, [x2]
  // CHECK-ERROR:             ^

  ldumaxalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalh h2, h3, [sp]
  // CHECK-ERROR:             ^

  ldumaxalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumaxalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  ldumin b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumin b0, b1, [x2]
  // CHECK-ERROR:          ^

  ldumin b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumin b2, b3, [sp]
  // CHECK-ERROR:          ^

  ldumin h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumin h0, h1, [x2]
  // CHECK-ERROR:          ^

  ldumin h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumin h2, h3, [sp]
  // CHECK-ERROR:          ^

  ldumin v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumin v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:          ^

  ldumina b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumina b0, b1, [x2]
  // CHECK-ERROR:           ^

  ldumina b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumina b2, b3, [sp]
  // CHECK-ERROR:           ^

  ldumina h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumina h0, h1, [x2]
  // CHECK-ERROR:           ^

  ldumina h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumina h2, h3, [sp]
  // CHECK-ERROR:           ^

  ldumina v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   ldumina v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  lduminl b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminl b0, b1, [x2]
  // CHECK-ERROR:           ^

  lduminl b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminl b2, b3, [sp]
  // CHECK-ERROR:           ^

  lduminl h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminl h0, h1, [x2]
  // CHECK-ERROR:           ^

  lduminl h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminl h2, h3, [sp]
  // CHECK-ERROR:           ^

  lduminl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminl v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  lduminal b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminal b0, b1, [x2]
  // CHECK-ERROR:            ^

  lduminal b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminal b2, b3, [sp]
  // CHECK-ERROR:            ^

  lduminal h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminal h0, h1, [x2]
  // CHECK-ERROR:            ^

  lduminal h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminal h2, h3, [sp]
  // CHECK-ERROR:            ^

  lduminal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminal v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  lduminb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminb b0, b1, [x2]
  // CHECK-ERROR:           ^

  lduminb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminb b2, b3, [sp]
  // CHECK-ERROR:           ^

  lduminb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminb h0, h1, [x2]
  // CHECK-ERROR:           ^

  lduminb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminb h2, h3, [sp]
  // CHECK-ERROR:           ^

  lduminb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  lduminh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminh b0, b1, [x2]
  // CHECK-ERROR:           ^

  lduminh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminh b2, b3, [sp]
  // CHECK-ERROR:           ^

  lduminh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminh h0, h1, [x2]
  // CHECK-ERROR:           ^

  lduminh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminh h2, h3, [sp]
  // CHECK-ERROR:           ^

  lduminh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:           ^

  lduminab b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminab b0, b1, [x2]
  // CHECK-ERROR:            ^

  lduminab b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminab b2, b3, [sp]
  // CHECK-ERROR:            ^

  lduminab h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminab h0, h1, [x2]
  // CHECK-ERROR:            ^

  lduminab h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminab h2, h3, [sp]
  // CHECK-ERROR:            ^

  lduminab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminab v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  lduminlb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlb b0, b1, [x2]
  // CHECK-ERROR:            ^

  lduminlb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlb b2, b3, [sp]
  // CHECK-ERROR:            ^

  lduminlb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlb h0, h1, [x2]
  // CHECK-ERROR:            ^

  lduminlb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlb h2, h3, [sp]
  // CHECK-ERROR:            ^

  lduminlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  lduminalb b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalb b0, b1, [x2]
  // CHECK-ERROR:             ^

  lduminalb b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalb b2, b3, [sp]
  // CHECK-ERROR:             ^

  lduminalb h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalb h0, h1, [x2]
  // CHECK-ERROR:             ^

  lduminalb h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalb h2, h3, [sp]
  // CHECK-ERROR:             ^

  lduminalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalb v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  lduminah b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminah b0, b1, [x2]
  // CHECK-ERROR:            ^

  lduminah b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminah b2, b3, [sp]
  // CHECK-ERROR:            ^

  lduminah h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminah h0, h1, [x2]
  // CHECK-ERROR:            ^

  lduminah h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminah h2, h3, [sp]
  // CHECK-ERROR:            ^

  lduminah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminah v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  lduminlh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlh b0, b1, [x2]
  // CHECK-ERROR:            ^

  lduminlh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlh b2, b3, [sp]
  // CHECK-ERROR:            ^

  lduminlh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlh h0, h1, [x2]
  // CHECK-ERROR:            ^

  lduminlh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlh h2, h3, [sp]
  // CHECK-ERROR:            ^

  lduminlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminlh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:            ^

  lduminalh b0, b1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalh b0, b1, [x2]
  // CHECK-ERROR:             ^

  lduminalh b2, b3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalh b2, b3, [sp]
  // CHECK-ERROR:             ^

  lduminalh h0, h1, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalh h0, h1, [x2]
  // CHECK-ERROR:             ^

  lduminalh h2, h3, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalh h2, h3, [sp]
  // CHECK-ERROR:             ^

  lduminalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   lduminalh v0.4h, v1.4h, v2.4h
  // CHECK-ERROR:             ^

  stadd b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd b0, [x2]
  // CHECK-ERROR:         ^

  stadd b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd b2, [sp]
  // CHECK-ERROR:         ^

  stadd h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd h0, [x2]
  // CHECK-ERROR:         ^

  stadd h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd h2, [sp]
  // CHECK-ERROR:         ^

  stadd v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd v0.4h, v2.4h
  // CHECK-ERROR:         ^

  staddl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl b0, [x2]
  // CHECK-ERROR:          ^

  staddl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl b2, [sp]
  // CHECK-ERROR:          ^

  staddl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl h0, [x2]
  // CHECK-ERROR:          ^

  staddl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl h2, [sp]
  // CHECK-ERROR:          ^

  staddl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl v0.4h, v2.4h
  // CHECK-ERROR:          ^

  staddb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddb b0, [x2]
  // CHECK-ERROR:          ^

  staddb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddb b2, [sp]
  // CHECK-ERROR:          ^

  staddb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddb h0, [x2]
  // CHECK-ERROR:          ^

  staddb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddb h2, [sp]
  // CHECK-ERROR:          ^

  staddb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddb v0.4h, v2.4h
  // CHECK-ERROR:          ^

  staddh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddh b0, [x2]
  // CHECK-ERROR:          ^

  staddh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddh b2, [sp]
  // CHECK-ERROR:          ^

  staddh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddh h0, [x2]
  // CHECK-ERROR:          ^

  staddh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddh h2, [sp]
  // CHECK-ERROR:          ^

  staddh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddh v0.4h, v2.4h
  // CHECK-ERROR:          ^

  staddlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlb b0, [x2]
  // CHECK-ERROR:           ^

  staddlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlb b2, [sp]
  // CHECK-ERROR:           ^

  staddlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlb h0, [x2]
  // CHECK-ERROR:           ^

  staddlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlb h2, [sp]
  // CHECK-ERROR:           ^

  staddlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  staddlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlh b0, [x2]
  // CHECK-ERROR:           ^

  staddlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlh b2, [sp]
  // CHECK-ERROR:           ^

  staddlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlh h0, [x2]
  // CHECK-ERROR:           ^

  staddlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlh h2, [sp]
  // CHECK-ERROR:           ^

  staddlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddlh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stadd b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd b0, [x2]
  // CHECK-ERROR:         ^

  stadd b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd b2, [sp]
  // CHECK-ERROR:         ^

  stadd h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd h0, [x2]
  // CHECK-ERROR:         ^

  stadd h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd h2, [sp]
  // CHECK-ERROR:         ^

  stadd v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stadd v0.4h, v2.4h
  // CHECK-ERROR:         ^

  staddl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl b0, [x2]
  // CHECK-ERROR:          ^

  staddl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl b2, [sp]
  // CHECK-ERROR:          ^

  staddl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl h0, [x2]
  // CHECK-ERROR:          ^

  staddl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl h2, [sp]
  // CHECK-ERROR:          ^

  staddl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   staddl v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stclr b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclr b0, [x2]
  // CHECK-ERROR:         ^

  stclr b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclr b2, [sp]
  // CHECK-ERROR:         ^

  stclr h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclr h0, [x2]
  // CHECK-ERROR:         ^

  stclr h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclr h2, [sp]
  // CHECK-ERROR:         ^

  stclr v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclr v0.4h, v2.4h
  // CHECK-ERROR:         ^

  stclrl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrl b0, [x2]
  // CHECK-ERROR:          ^

  stclrl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrl b2, [sp]
  // CHECK-ERROR:          ^

  stclrl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrl h0, [x2]
  // CHECK-ERROR:          ^

  stclrl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrl h2, [sp]
  // CHECK-ERROR:          ^

  stclrl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrl v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stclrb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrb b0, [x2]
  // CHECK-ERROR:          ^

  stclrb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrb b2, [sp]
  // CHECK-ERROR:          ^

  stclrb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrb h0, [x2]
  // CHECK-ERROR:          ^

  stclrb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrb h2, [sp]
  // CHECK-ERROR:          ^

  stclrb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrb v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stclrh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrh b0, [x2]
  // CHECK-ERROR:          ^

  stclrh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrh b2, [sp]
  // CHECK-ERROR:          ^

  stclrh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrh h0, [x2]
  // CHECK-ERROR:          ^

  stclrh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrh h2, [sp]
  // CHECK-ERROR:          ^

  stclrh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrh v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stclrlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlb b0, [x2]
  // CHECK-ERROR:           ^

  stclrlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlb b2, [sp]
  // CHECK-ERROR:           ^

  stclrlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlb h0, [x2]
  // CHECK-ERROR:           ^

  stclrlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlb h2, [sp]
  // CHECK-ERROR:           ^

  stclrlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stclrlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlh b0, [x2]
  // CHECK-ERROR:           ^

  stclrlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlh b2, [sp]
  // CHECK-ERROR:           ^

  stclrlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlh h0, [x2]
  // CHECK-ERROR:           ^

  stclrlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlh h2, [sp]
  // CHECK-ERROR:           ^

  stclrlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stclrlh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  steor b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steor b0, [x2]
  // CHECK-ERROR:         ^

  steor b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steor b2, [sp]
  // CHECK-ERROR:         ^

  steor h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steor h0, [x2]
  // CHECK-ERROR:         ^

  steor h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steor h2, [sp]
  // CHECK-ERROR:         ^

  steor v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steor v0.4h, v2.4h
  // CHECK-ERROR:         ^

  steorl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorl b0, [x2]
  // CHECK-ERROR:          ^

  steorl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorl b2, [sp]
  // CHECK-ERROR:          ^

  steorl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorl h0, [x2]
  // CHECK-ERROR:          ^

  steorl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorl h2, [sp]
  // CHECK-ERROR:          ^

  steorl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorl v0.4h, v2.4h
  // CHECK-ERROR:          ^

  steorb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorb b0, [x2]
  // CHECK-ERROR:          ^

  steorb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorb b2, [sp]
  // CHECK-ERROR:          ^

  steorb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorb h0, [x2]
  // CHECK-ERROR:          ^

  steorb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorb h2, [sp]
  // CHECK-ERROR:          ^

  steorb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorb v0.4h, v2.4h
  // CHECK-ERROR:          ^

  steorh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorh b0, [x2]
  // CHECK-ERROR:          ^

  steorh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorh b2, [sp]
  // CHECK-ERROR:          ^

  steorh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorh h0, [x2]
  // CHECK-ERROR:          ^

  steorh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorh h2, [sp]
  // CHECK-ERROR:          ^

  steorh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorh v0.4h, v2.4h
  // CHECK-ERROR:          ^

  steorlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlb b0, [x2]
  // CHECK-ERROR:           ^

  steorlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlb b2, [sp]
  // CHECK-ERROR:           ^

  steorlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlb h0, [x2]
  // CHECK-ERROR:           ^

  steorlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlb h2, [sp]
  // CHECK-ERROR:           ^

  steorlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  steorlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlh b0, [x2]
  // CHECK-ERROR:           ^

  steorlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlh b2, [sp]
  // CHECK-ERROR:           ^

  steorlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlh h0, [x2]
  // CHECK-ERROR:           ^

  steorlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlh h2, [sp]
  // CHECK-ERROR:           ^

  steorlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   steorlh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stset b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stset b0, [x2]
  // CHECK-ERROR:         ^

  stset b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stset b2, [sp]
  // CHECK-ERROR:         ^

  stset h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stset h0, [x2]
  // CHECK-ERROR:         ^

  stset h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stset h2, [sp]
  // CHECK-ERROR:         ^

  stset v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stset v0.4h, v2.4h
  // CHECK-ERROR:         ^

  stsetl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetl b0, [x2]
  // CHECK-ERROR:          ^

  stsetl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetl b2, [sp]
  // CHECK-ERROR:          ^

  stsetl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetl h0, [x2]
  // CHECK-ERROR:          ^

  stsetl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetl h2, [sp]
  // CHECK-ERROR:          ^

  stsetl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetl v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stsetb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetb b0, [x2]
  // CHECK-ERROR:          ^

  stsetb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetb b2, [sp]
  // CHECK-ERROR:          ^

  stsetb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetb h0, [x2]
  // CHECK-ERROR:          ^

  stsetb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetb h2, [sp]
  // CHECK-ERROR:          ^

  stsetb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetb v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stseth b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stseth b0, [x2]
  // CHECK-ERROR:          ^

  stseth b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stseth b2, [sp]
  // CHECK-ERROR:          ^

  stseth h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stseth h0, [x2]
  // CHECK-ERROR:          ^

  stseth h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stseth h2, [sp]
  // CHECK-ERROR:          ^

  stseth v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stseth v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stsetlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlb b0, [x2]
  // CHECK-ERROR:           ^

  stsetlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlb b2, [sp]
  // CHECK-ERROR:           ^

  stsetlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlb h0, [x2]
  // CHECK-ERROR:           ^

  stsetlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlb h2, [sp]
  // CHECK-ERROR:           ^

  stsetlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsetlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlh b0, [x2]
  // CHECK-ERROR:           ^

  stsetlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlh b2, [sp]
  // CHECK-ERROR:           ^

  stsetlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlh h0, [x2]
  // CHECK-ERROR:           ^

  stsetlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlh h2, [sp]
  // CHECK-ERROR:           ^

  stsetlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsetlh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsmax b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmax b0, [x2]
  // CHECK-ERROR:          ^

  stsmax b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmax b2, [sp]
  // CHECK-ERROR:          ^

  stsmax h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmax h0, [x2]
  // CHECK-ERROR:          ^

  stsmax h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmax h2, [sp]
  // CHECK-ERROR:          ^

  stsmax v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmax v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stsmaxl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxl b0, [x2]
  // CHECK-ERROR:           ^

  stsmaxl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxl b2, [sp]
  // CHECK-ERROR:           ^

  stsmaxl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxl h0, [x2]
  // CHECK-ERROR:           ^

  stsmaxl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxl h2, [sp]
  // CHECK-ERROR:           ^

  stsmaxl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxl v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsmaxb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxb b0, [x2]
  // CHECK-ERROR:           ^

  stsmaxb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxb b2, [sp]
  // CHECK-ERROR:           ^

  stsmaxb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxb h0, [x2]
  // CHECK-ERROR:           ^

  stsmaxb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxb h2, [sp]
  // CHECK-ERROR:           ^

  stsmaxb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsmaxh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxh b0, [x2]
  // CHECK-ERROR:           ^

  stsmaxh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxh b2, [sp]
  // CHECK-ERROR:           ^

  stsmaxh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxh h0, [x2]
  // CHECK-ERROR:           ^

  stsmaxh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxh h2, [sp]
  // CHECK-ERROR:           ^

  stsmaxh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsmaxlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlb b0, [x2]
  // CHECK-ERROR:            ^

  stsmaxlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlb b2, [sp]
  // CHECK-ERROR:            ^

  stsmaxlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlb h0, [x2]
  // CHECK-ERROR:            ^

  stsmaxlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlb h2, [sp]
  // CHECK-ERROR:            ^

  stsmaxlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlb v0.4h, v2.4h
  // CHECK-ERROR:            ^

  stsmaxlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlh b0, [x2]
  // CHECK-ERROR:            ^

  stsmaxlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlh b2, [sp]
  // CHECK-ERROR:            ^

  stsmaxlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlh h0, [x2]
  // CHECK-ERROR:            ^

  stsmaxlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlh h2, [sp]
  // CHECK-ERROR:            ^

  stsmaxlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmaxlh v0.4h, v2.4h
  // CHECK-ERROR:            ^

  stsmin b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmin b0, [x2]
  // CHECK-ERROR:          ^

  stsmin b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmin b2, [sp]
  // CHECK-ERROR:          ^

  stsmin h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmin h0, [x2]
  // CHECK-ERROR:          ^

  stsmin h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmin h2, [sp]
  // CHECK-ERROR:          ^

  stsmin v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsmin v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stsminl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminl b0, [x2]
  // CHECK-ERROR:           ^

  stsminl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminl b2, [sp]
  // CHECK-ERROR:           ^

  stsminl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminl h0, [x2]
  // CHECK-ERROR:           ^

  stsminl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminl h2, [sp]
  // CHECK-ERROR:           ^

  stsminl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminl v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsminb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminb b0, [x2]
  // CHECK-ERROR:           ^

  stsminb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminb b2, [sp]
  // CHECK-ERROR:           ^

  stsminb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminb h0, [x2]
  // CHECK-ERROR:           ^

  stsminb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminb h2, [sp]
  // CHECK-ERROR:           ^

  stsminb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsminh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminh b0, [x2]
  // CHECK-ERROR:           ^

  stsminh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminh b2, [sp]
  // CHECK-ERROR:           ^

  stsminh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminh h0, [x2]
  // CHECK-ERROR:           ^

  stsminh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminh h2, [sp]
  // CHECK-ERROR:           ^

  stsminh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stsminlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlb b0, [x2]
  // CHECK-ERROR:            ^

  stsminlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlb b2, [sp]
  // CHECK-ERROR:            ^

  stsminlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlb h0, [x2]
  // CHECK-ERROR:            ^

  stsminlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlb h2, [sp]
  // CHECK-ERROR:            ^

  stsminlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlb v0.4h, v2.4h
  // CHECK-ERROR:            ^

  stsminlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlh b0, [x2]
  // CHECK-ERROR:            ^

  stsminlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlh b2, [sp]
  // CHECK-ERROR:            ^

  stsminlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlh h0, [x2]
  // CHECK-ERROR:            ^

  stsminlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlh h2, [sp]
  // CHECK-ERROR:            ^

  stsminlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stsminlh v0.4h, v2.4h
  // CHECK-ERROR:            ^

  stumax b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumax b0, [x2]
  // CHECK-ERROR:          ^

  stumax b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumax b2, [sp]
  // CHECK-ERROR:          ^

  stumax h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumax h0, [x2]
  // CHECK-ERROR:          ^

  stumax h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumax h2, [sp]
  // CHECK-ERROR:          ^

  stumax v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumax v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stumaxl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxl b0, [x2]
  // CHECK-ERROR:           ^

  stumaxl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxl b2, [sp]
  // CHECK-ERROR:           ^

  stumaxl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxl h0, [x2]
  // CHECK-ERROR:           ^

  stumaxl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxl h2, [sp]
  // CHECK-ERROR:           ^

  stumaxl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxl v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stumaxb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxb b0, [x2]
  // CHECK-ERROR:           ^

  stumaxb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxb b2, [sp]
  // CHECK-ERROR:           ^

  stumaxb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxb h0, [x2]
  // CHECK-ERROR:           ^

  stumaxb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxb h2, [sp]
  // CHECK-ERROR:           ^

  stumaxb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stumaxh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxh b0, [x2]
  // CHECK-ERROR:           ^

  stumaxh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxh b2, [sp]
  // CHECK-ERROR:           ^

  stumaxh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxh h0, [x2]
  // CHECK-ERROR:           ^

  stumaxh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxh h2, [sp]
  // CHECK-ERROR:           ^

  stumaxh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxh v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stumaxlb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlb b0, [x2]
  // CHECK-ERROR:            ^

  stumaxlb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlb b2, [sp]
  // CHECK-ERROR:            ^

  stumaxlb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlb h0, [x2]
  // CHECK-ERROR:            ^

  stumaxlb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlb h2, [sp]
  // CHECK-ERROR:            ^

  stumaxlb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlb v0.4h, v2.4h
  // CHECK-ERROR:            ^

  stumaxlh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlh b0, [x2]
  // CHECK-ERROR:            ^

  stumaxlh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlh b2, [sp]
  // CHECK-ERROR:            ^

  stumaxlh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlh h0, [x2]
  // CHECK-ERROR:            ^

  stumaxlh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlh h2, [sp]
  // CHECK-ERROR:            ^

  stumaxlh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumaxlh v0.4h, v2.4h
  // CHECK-ERROR:            ^

  stumin b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumin b0, [x2]
  // CHECK-ERROR:          ^

  stumin b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumin b2, [sp]
  // CHECK-ERROR:          ^

  stumin h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumin h0, [x2]
  // CHECK-ERROR:          ^

  stumin h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumin h2, [sp]
  // CHECK-ERROR:          ^

  stumin v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stumin v0.4h, v2.4h
  // CHECK-ERROR:          ^

  stuminl b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminl b0, [x2]
  // CHECK-ERROR:           ^

  stuminl b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminl b2, [sp]
  // CHECK-ERROR:           ^

  stuminl h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminl h0, [x2]
  // CHECK-ERROR:           ^

  stuminl h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminl h2, [sp]
  // CHECK-ERROR:           ^

  stuminl v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminl v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stuminb b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminb b0, [x2]
  // CHECK-ERROR:           ^

  stuminb b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminb b2, [sp]
  // CHECK-ERROR:           ^

  stuminb h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminb h0, [x2]
  // CHECK-ERROR:           ^

  stuminb h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminb h2, [sp]
  // CHECK-ERROR:           ^

  stuminb v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminb v0.4h, v2.4h
  // CHECK-ERROR:           ^

  stuminh b0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminh b0, [x2]
  // CHECK-ERROR:           ^

  stuminh b2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminh b2, [sp]
  // CHECK-ERROR:           ^

  stuminh h0, [x2]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminh h0, [x2]
  // CHECK-ERROR:           ^

  stuminh h2, [sp]
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminh h2, [sp]
  // CHECK-ERROR:           ^

  stuminh v0.4h, v2.4h
  // CHECK-ERROR: error: invalid operand for instruction
  // CHECK-ERROR:   stuminh v0.4h, v2.4h
  // CHECK-ERROR:           ^

