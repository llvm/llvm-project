// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.1a -show-encoding < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERROR <%t %s
  .text

  //8 bits
  casb   w0, w1, [x2]
  casab  w0, w1, [x2]
  caslb  w0, w1, [x2]
  casalb   w0, w1, [x2]

//CHECK:  casb   w0, w1, [x2]        //      encoding: [0x41,0x7c,0xa0,0x08]
//CHECK:  casab  w0, w1, [x2]        //      encoding: [0x41,0x7c,0xe0,0x08]
//CHECK:  caslb   w0, w1, [x2]       //      encoding: [0x41,0xfc,0xa0,0x08]
//CHECK:  casalb   w0, w1, [x2]      //      encoding: [0x41,0xfc,0xe0,0x08]

  casb w0, w1, [w2]
  casalb x0, x1, [x2]
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   casb w0, w1, [w2]
//CHECK-ERROR:                 ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   casalb x0, x1, [x2]
//CHECK-ERROR:          ^

  //16 bits
  cash   w0, w1, [x2]
  casah  w0, w1, [x2]
  caslh  w0, w1, [x2]
  casalh   w0, w1, [x2]

//CHECK:  cash   w0, w1, [x2]        //      encoding: [0x41,0x7c,0xa0,0x48]
//CHECK:  casah  w0, w1, [x2]        //      encoding: [0x41,0x7c,0xe0,0x48]
//CHECK:  caslh   w0, w1, [x2]       //      encoding: [0x41,0xfc,0xa0,0x48]
//CHECK:  casalh   w0, w1, [x2]      //      encoding: [0x41,0xfc,0xe0,0x48]

  //32 bits
  cas   w0, w1, [x2]
  casa  w0, w1, [x2]
  casl  w0, w1, [x2]
  casal   w0, w1, [x2]

//CHECK:  cas   w0, w1, [x2]        //      encoding: [0x41,0x7c,0xa0,0x88]
//CHECK:  casa  w0, w1, [x2]        //      encoding: [0x41,0x7c,0xe0,0x88]
//CHECK:  casl   w0, w1, [x2]       //      encoding: [0x41,0xfc,0xa0,0x88]
//CHECK:  casal   w0, w1, [x2]      //      encoding: [0x41,0xfc,0xe0,0x88]

  cas   w0, w1, [w2]
  casl  w0, x1, [x2]

//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   cas   w0, w1, [w2]
//CHECK-ERROR:                  ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   casl  w0, x1, [x2]
//CHECK-ERROR:             ^

  //64 bits
  cas   x0, x1, [x2]
  casa  x0, x1, [x2]
  casl   x0, x1, [x2]
  casal   x0, x1, [x2]

//CHECK:  cas   x0, x1, [x2]        //      encoding: [0x41,0x7c,0xa0,0xc8]
//CHECK:  casa  x0, x1, [x2]        //      encoding: [0x41,0x7c,0xe0,0xc8]
//CHECK:  casl   x0, x1, [x2]       //      encoding: [0x41,0xfc,0xa0,0xc8]
//CHECK:  casal   x0, x1, [x2]      //      encoding: [0x41,0xfc,0xe0,0xc8]

  casa   x0, x1, [w2]
  casal  x0, w1, [x2]

//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   casa   x0, x1, [w2]
//CHECK-ERROR:                   ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   casal  x0, w1, [x2]
//CHECK-ERROR:              ^

  // LD<OP> intructions
  ldadda x0, x1, [x2]
  ldclrl x0, x1, [x2]
  ldeoral x0, x1, [x2]
  ldset x0, x1, [x2]
  ldsmaxa w0, w1, [x2]
  ldsminlb w0, w1, [x2]
  ldumaxalh w0, w1, [x2]
  ldumin w0, w1, [x2]
  ldsminb w2, w3, [x5]
//CHECK: ldadda     x0, x1, [x2]  // encoding: [0x41,0x00,0xa0,0xf8]
//CHECK: ldclrl     x0, x1, [x2]  // encoding: [0x41,0x10,0x60,0xf8]
//CHECK: ldeoral    x0, x1, [x2]  // encoding: [0x41,0x20,0xe0,0xf8]
//CHECK: ldset      x0, x1, [x2]  // encoding: [0x41,0x30,0x20,0xf8]
//CHECK: ldsmaxa    w0, w1, [x2]  // encoding: [0x41,0x40,0xa0,0xb8]
//CHECK: ldsminlb   w0, w1, [x2]  // encoding: [0x41,0x50,0x60,0x38]
//CHECK: ldumaxalh  w0, w1, [x2]  // encoding: [0x41,0x60,0xe0,0x78]
//CHECK: ldumin     w0, w1, [x2]  // encoding: [0x41,0x70,0x20,0xb8]
//CHECK: ldsminb    w2, w3, [x5]  // encoding: [0xa3,0x50,0x22,0x38]

  // ST<OP> intructions: aliases to LD<OP>
  stADDlb w0, [x2]
  stclrlh w0, [x2]
  steorl  w0, [x2]
  stsetl  x0, [x2]
  stsmaxb  w0, [x2]
  stsminh  w0, [x2]
  stumax   w0, [x2]
  stumin   x0, [x2]
  stsminl x29, [sp]
//CHECK: staddlb    w0, [x2]  // encoding: [0x5f,0x00,0x60,0x38]
//CHECK: stclrlh    w0, [x2]  // encoding: [0x5f,0x10,0x60,0x78]
//CHECK: steorl     w0, [x2]  // encoding: [0x5f,0x20,0x60,0xb8]
//CHECK: stsetl     x0, [x2]  // encoding: [0x5f,0x30,0x60,0xf8]
//CHECK: stsmaxb     w0, [x2]  // encoding: [0x5f,0x40,0x20,0x38]
//CHECK: stsminh     w0, [x2]  // encoding: [0x5f,0x50,0x20,0x78]
//CHECK: stumax      w0, [x2]  // encoding: [0x5f,0x60,0x20,0xb8]
//CHECK: stumin      x0, [x2]  // encoding: [0x5f,0x70,0x20,0xf8]
//CHECK: stsminl     x29, [sp] // encoding: [0xff,0x53,0x7d,0xf8]


  ldsmax x0, x1, [w2]
  ldeorl w0, w1, [w2]
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   ldsmax x0, x1, [w2]
//CHECK-ERROR:                   ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   ldeorl w0, w1, [w2]
//CHECK-ERROR:                   ^

  //SWP instruction
  swp   x0, x1, [x2]
  swpb  w0, w1, [x2]
  swplh w0, w1, [x2]
  swpal x0, x1, [sp]
//CHECK: swp   x0, x1, [x2]       // encoding: [0x41,0x80,0x20,0xf8]
//CHECK: swpb  w0, w1, [x2]       // encoding: [0x41,0x80,0x20,0x38]
//CHECK: swplh w0, w1, [x2]       // encoding: [0x41,0x80,0x60,0x78]
//CHECK: swpal x0, x1, [sp]       // encoding: [0xe1,0x83,0xe0,0xf8]

  swp   x0, x1, [w2]
  swp   x0, x1, [xzr]
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   swp   x0, x1, [w2]
//CHECK-ERROR:                  ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   swp   x0, x1, [xzr]
//CHECK-ERROR:                  ^

  //CASP instruction
  casp x0, x1, x2, x3, [x4]
  casp w0, w1, w2, w3, [x4]
//CHECK: casp x0, x1, x2, x3, [x4]      // encoding: [0x82,0x7c,0x20,0x48]
//CHECK: casp w0, w1, w2, w3, [x4]      // encoding: [0x82,0x7c,0x20,0x08]

  casp x1, x2, x4, x5, [x6]
  casp x0, x1, x3, x4, [x5]
  casp x0, x2, x4, x5, [x6]
  casp x0, x1, x2, x4, [x5]
  casp x0, w1, x2, x3, [x5]
  casp w0, x1, x2, x3, [x5]
  casp w0, x1, w2, w3, [x5]
  casp x0, x1, w2, w3, [x5]
//CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp x1, x2, x4, x5, [x6]
//CHECK-ERROR:       ^
//CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp x0, x1, x3, x4, [x5]
//CHECK-ERROR:               ^
//CHECK-ERROR: error:  expected second odd register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp x0, x2, x4, x5, [x6]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: expected second odd register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp x0, x1, x2, x4, [x5]
//CHECK-ERROR:                   ^
//CHECK-ERROR: error: expected second odd register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp x0, w1, x2, x3, [x5]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: expected second odd register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp w0, x1, x2, x3, [x5]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: expected second odd register of a consecutive same-size even/odd register pair
//CHECK-ERROR:  casp w0, x1, w2, w3, [x5]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:  casp x0, x1, w2, w3, [x5]
//CHECK-ERROR:               ^
