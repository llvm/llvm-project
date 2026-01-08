// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lsui < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lsui < %s \
// RUN:        | llvm-objdump -d --mattr=+lsui --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lsui < %s \
// RUN:        | llvm-objdump -d --mattr=-lsui --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lsui < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+lsui -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



//------------------------------------------------------------------------------
// Unprivileged load/store operations
//------------------------------------------------------------------------------
ldtxr x9, [sp]
// CHECK-INST: ldtxr x9, [sp]
// CHECK-ENCODING: encoding: [0xe9,0x7f,0x5f,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c95f7fe9 <unknown>

ldtxr x9, [sp, #0]
// CHECK-INST: ldtxr x9, [sp]
// CHECK-ENCODING: encoding: [0xe9,0x7f,0x5f,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c95f7fe9 <unknown>

ldtxr x10, [x11]
// CHECK-INST: ldtxr x10, [x11]
// CHECK-ENCODING: encoding: [0x6a,0x7d,0x5f,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c95f7d6a <unknown>

ldtxr x10, [x11, #0]
// CHECK-INST: ldtxr x10, [x11]
// CHECK-ENCODING: encoding: [0x6a,0x7d,0x5f,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c95f7d6a <unknown>

ldatxr x9, [sp]
// CHECK-INST: ldatxr x9, [sp]
// CHECK-ENCODING: encoding: [0xe9,0xff,0x5f,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c95fffe9 <unknown>

ldatxr x10, [x11]
// CHECK-INST: ldatxr x10, [x11]
// CHECK-ENCODING: encoding: [0x6a,0xfd,0x5f,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c95ffd6a <unknown>

sttxr wzr, w4, [sp]
// CHECK-INST: sttxr wzr, w4, [sp]
// CHECK-ENCODING: encoding: [0xe4,0x7f,0x1f,0x89]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  891f7fe4 <unknown>

sttxr wzr, w4, [sp, #0]
// CHECK-INST: sttxr wzr, w4, [sp]
// CHECK-ENCODING: encoding: [0xe4,0x7f,0x1f,0x89]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  891f7fe4 <unknown>

sttxr w5, x6, [x7]
// CHECK-INST: sttxr w5, x6, [x7]
// CHECK-ENCODING: encoding: [0xe6,0x7c,0x05,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9057ce6 <unknown>

sttxr w5, x6, [x7, #0]
// CHECK-INST: sttxr w5, x6, [x7]
// CHECK-ENCODING: encoding: [0xe6,0x7c,0x05,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9057ce6 <unknown>

stltxr w2, w4, [sp]
// CHECK-INST: stltxr w2, w4, [sp]
// CHECK-ENCODING: encoding: [0xe4,0xff,0x02,0x89]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  8902ffe4 <unknown>

stltxr w5, x6, [x7]
// CHECK-INST: stltxr w5, x6, [x7]
// CHECK-ENCODING: encoding: [0xe6,0xfc,0x05,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c905fce6 <unknown>

//------------------------------------------------------------------------------
// Unprivileged load/store register pair (offset)
//------------------------------------------------------------------------------

ldtp x21, x29, [x2, #504]
// CHECK-INST: ldtp x21, x29, [x2, #504]
// CHECK-ENCODING: encoding: [0x55,0xf4,0x5f,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e95ff455 <unknown>

ldtp x22, x23, [x3, #-512]
// CHECK-INST: ldtp x22, x23, [x3, #-512]
// CHECK-ENCODING: encoding: [0x76,0x5c,0x60,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e9605c76 <unknown>

ldtp x24, x25, [x4, #8]
// CHECK-INST: ldtp x24, x25, [x4, #8]
// CHECK-ENCODING: encoding: [0x98,0xe4,0x40,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e940e498 <unknown>

sttp x3, x5, [sp], #16
// CHECK-INST: sttp x3, x5, [sp], #16
// CHECK-ENCODING: encoding: [0xe3,0x17,0x81,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e88117e3 <unknown>

sttp x3, x5, [sp, #8]!
// CHECK-INST: sttp x3, x5, [sp, #8]!
// CHECK-ENCODING: encoding: [0xe3,0x97,0x80,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e98097e3 <unknown>

sttp q3, q5, [sp]
// CHECK-INST: sttp q3, q5, [sp]
// CHECK-ENCODING: encoding: [0xe3,0x17,0x00,0xed]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ed0017e3 <unknown>

sttp q17, q19, [sp, #1008]
// CHECK-INST: sttp q17, q19, [sp, #1008]
// CHECK-ENCODING: encoding: [0xf1,0xcf,0x1f,0xed]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ed1fcff1 <unknown>

//------------------------------------------------------------------------------
// Load/store register pair (post-indexed)
//------------------------------------------------------------------------------

ldtp x21, x29, [x2], #504
// CHECK-INST: ldtp x21, x29, [x2], #504
// CHECK-ENCODING: encoding: [0x55,0xf4,0xdf,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e8dff455 <unknown>

ldtp x22, x23, [x3], #-512
// CHECK-INST: ldtp x22, x23, [x3], #-512
// CHECK-ENCODING: encoding: [0x76,0x5c,0xe0,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e8e05c76 <unknown>

ldtp x24, x25, [x4], #8
// CHECK-INST: ldtp x24, x25, [x4], #8
// CHECK-ENCODING: encoding: [0x98,0xe4,0xc0,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e8c0e498 <unknown>

sttp q3, q5, [sp], #0
// CHECK-INST: sttp q3, q5, [sp], #0
// CHECK-ENCODING: encoding: [0xe3,0x17,0x80,0xec]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ec8017e3 <unknown>

sttp q17, q19, [sp], #1008
// CHECK-INST: sttp q17, q19, [sp], #1008
// CHECK-ENCODING: encoding: [0xf1,0xcf,0x9f,0xec]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ec9fcff1 <unknown>

ldtp q23, q29, [x1], #-1024
// CHECK-INST: ldtp q23, q29, [x1], #-1024
// CHECK-ENCODING: encoding: [0x37,0x74,0xe0,0xec]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ece07437 <unknown>

//------------------------------------------------------------------------------
// Load/store register pair (pre-indexed)
//------------------------------------------------------------------------------
ldtp x21, x29, [x2, #504]!
// CHECK-INST: ldtp x21, x29, [x2, #504]!
// CHECK-ENCODING: encoding: [0x55,0xf4,0xdf,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e9dff455 <unknown>

ldtp x22, x23, [x3, #-512]!
// CHECK-INST: ldtp x22, x23, [x3, #-512]!
// CHECK-ENCODING: encoding: [0x76,0x5c,0xe0,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e9e05c76 <unknown>

ldtp x24, x25, [x4, #8]!
// CHECK-INST: ldtp x24, x25, [x4, #8]!
// CHECK-ENCODING: encoding: [0x98,0xe4,0xc0,0xe9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e9c0e498 <unknown>

sttp q3, q5, [sp, #0]!
// CHECK-INST: sttp q3, q5, [sp, #0]!
// CHECK-ENCODING: encoding: [0xe3,0x17,0x80,0xed]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ed8017e3 <unknown>

sttp q17, q19, [sp, #1008]!
// CHECK-INST: sttp q17, q19, [sp, #1008]!
// CHECK-ENCODING: encoding: [0xf1,0xcf,0x9f,0xed]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ed9fcff1 <unknown>

ldtp q23, q29, [x1, #-1024]!
// CHECK-INST: ldtp q23, q29, [x1, #-1024]!
// CHECK-ENCODING: encoding: [0x37,0x74,0xe0,0xed]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ede07437 <unknown>

//------------------------------------------------------------------------------
// CAS(P)T instructions
//------------------------------------------------------------------------------
  //64 bits
  cast x0, x1, [x2]
// CHECK-INST: cast x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x41,0x7c,0x80,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9807c41 <unknown>

  cast x0, x1, [sp, #0]
// CHECK-INST: cast x0, x1, [sp]
// CHECK-ENCODING: encoding: [0xe1,0x7f,0x80,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9807fe1 <unknown>

  casat x0, x1, [x2]
// CHECK-INST: casat x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x41,0x7c,0xc0,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9c07c41 <unknown>

  casat x0, x1, [sp, #0]
// CHECK-INST: casat x0, x1, [sp]
// CHECK-ENCODING: encoding: [0xe1,0x7f,0xc0,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9c07fe1 <unknown>

  casalt x0, x1, [x2]
// CHECK-INST: casalt x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x41,0xfc,0xc0,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9c0fc41 <unknown>

  casalt x0, x1, [sp, #0]
// CHECK-INST: casalt x0, x1, [sp]
// CHECK-ENCODING: encoding: [0xe1,0xff,0xc0,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c9c0ffe1 <unknown>

  caslt x0, x1, [x2]
// CHECK-INST: caslt x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x41,0xfc,0x80,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c980fc41 <unknown>

  caslt x0, x1, [sp, #0]
// CHECK-INST: caslt x0, x1, [sp]
// CHECK-ENCODING: encoding: [0xe1,0xff,0x80,0xc9]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  c980ffe1 <unknown>

  //CASP instruction
caspt x0, x1, x2, x3, [x4]
// CHECK-INST: caspt x0, x1, x2, x3, [x4]
// CHECK-ENCODING: encoding: [0x82,0x7c,0x80,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  49807c82 <unknown>

caspt x0, x1, x2, x3, [sp, #0]
// CHECK-INST: caspt x0, x1, x2, x3, [sp]
// CHECK-ENCODING: encoding: [0xe2,0x7f,0x80,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  49807fe2 <unknown>

caspat x0, x1, x2, x3, [x4]
// CHECK-INST: caspat x0, x1, x2, x3, [x4]
// CHECK-ENCODING: encoding: [0x82,0x7c,0xc0,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  49c07c82 <unknown>

caspat x0, x1, x2, x3, [sp, #0]
// CHECK-INST: caspat x0, x1, x2, x3, [sp]
// CHECK-ENCODING: encoding: [0xe2,0x7f,0xc0,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  49c07fe2 <unknown>

casplt x0, x1, x2, x3, [x4]
// CHECK-INST: casplt x0, x1, x2, x3, [x4]
// CHECK-ENCODING: encoding: [0x82,0xfc,0x80,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  4980fc82 <unknown>

casplt x0, x1, x2, x3, [sp, #0]
// CHECK-INST: casplt x0, x1, x2, x3, [sp]
// CHECK-ENCODING: encoding: [0xe2,0xff,0x80,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  4980ffe2 <unknown>

caspalt x0, x1, x2, x3, [x4]
// CHECK-INST: caspalt x0, x1, x2, x3, [x4]
// CHECK-ENCODING: encoding: [0x82,0xfc,0xc0,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  49c0fc82 <unknown>

caspalt x0, x1, x2, x3, [sp, #0]
// CHECK-INST: caspalt x0, x1, x2, x3, [sp]
// CHECK-ENCODING: encoding: [0xe2,0xff,0xc0,0x49]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  49c0ffe2 <unknown>

//------------------------------------------------------------------------------
// SWP(A|L)T instructions
//------------------------------------------------------------------------------
swpt w7, wzr, [x5]
// CHECK-INST: swpt w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x84,0x27,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192784bf <unknown>

swpt x9, xzr, [sp]
// CHECK-INST: swpt x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x87,0x29,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592987ff <unknown>

swpta w7, wzr, [x5]
// CHECK-INST: swpta w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x84,0xa7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19a784bf <unknown>

swpta x9, xzr, [sp]
// CHECK-INST: swpta x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x87,0xa9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59a987ff <unknown>

swptl w7, wzr, [x5]
// CHECK-INST: swptl w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x84,0x67,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196784bf <unknown>

swptl x9, xzr, [sp]
// CHECK-INST: swptl x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x87,0x69,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596987ff <unknown>

swptal w7, wzr, [x5]
// CHECK-INST: swptal w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x84,0xe7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19e784bf <unknown>

swptal x9, xzr, [sp]
// CHECK-INST: swptal x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x87,0xe9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59e987ff <unknown>

//------------------------------------------------------------------------------
// LD{ADD|CLR|SET)(A|L|AL)T instructions
//------------------------------------------------------------------------------

ldtadd w7, wzr, [x5]
// CHECK-INST: sttadd w7, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x04,0x27,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192704bf <unknown>

ldtadd x9, xzr, [sp]
// CHECK-INST: sttadd x9, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0x29,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592907ff <unknown>

ldtadda w7, wzr, [x5]
// CHECK-INST: ldtadda w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x04,0xa7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19a704bf <unknown>

ldtadda x9, xzr, [sp]
// CHECK-INST: ldtadda x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0xa9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59a907ff <unknown>

ldtaddl w7, wzr, [x5]
// CHECK-INST: sttaddl w7, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x04,0x67,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196704bf <unknown>

ldtaddl x9, xzr, [sp]
// CHECK-INST: sttaddl x9, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0x69,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596907ff <unknown>

ldtaddal w7, wzr, [x5]
// CHECK-INST: ldtaddal w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x04,0xe7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19e704bf <unknown>

ldtaddal x9, xzr, [sp]
// CHECK-INST: ldtaddal x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0xe9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59e907ff <unknown>

ldtclr w7, wzr, [x5]
// CHECK-INST: sttclr w7, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x14,0x27,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192714bf <unknown>

ldtclr x9, xzr, [sp]
// CHECK-INST: sttclr x9, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0x29,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592917ff <unknown>

ldtclrl w7, wzr, [x5]
// CHECK-INST: sttclrl w7, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x14,0x67,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196714bf <unknown>

ldtclrl x9, xzr, [sp]
// CHECK-INST: sttclrl x9, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0x69,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596917ff <unknown>

ldtclra w7, wzr, [x5]
// CHECK-INST: ldtclra w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x14,0xa7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19a714bf <unknown>

ldtclra x9, xzr, [sp]
// CHECK-INST: ldtclra x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0xa9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59a917ff <unknown>

ldtclral w7, wzr, [x5]
// CHECK-INST: ldtclral w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x14,0xe7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19e714bf <unknown>

ldtclral x9, xzr, [sp]
// CHECK-INST: ldtclral x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0xe9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59e917ff <unknown>

ldtset w7, wzr, [x5]
// CHECK-INST: sttset w7, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x34,0x27,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192734bf <unknown>

ldtset x9, xzr, [sp]
// CHECK-INST: sttset x9, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0x29,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592937ff <unknown>

ldtsetl w7, wzr, [x5]
// CHECK-INST: sttsetl w7, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x34,0x67,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196734bf <unknown>

ldtsetl x9, xzr, [sp]
// CHECK-INST: sttsetl x9, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0x69,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596937ff <unknown>

ldtseta w7, wzr, [x5]
// CHECK-INST: ldtseta w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x34,0xa7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19a734bf <unknown>

ldtseta x9, xzr, [sp]
// CHECK-INST: ldtseta x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0xa9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59a937ff <unknown>

ldtsetal w7, wzr, [x5]
// CHECK-INST: ldtsetal w7, wzr, [x5]
// CHECK-ENCODING: encoding: [0xbf,0x34,0xe7,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  19e734bf <unknown>

ldtsetal x9, xzr, [sp]
// CHECK-INST: ldtsetal x9, xzr, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0xe9,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  59e937ff <unknown>

//------------------------------------------------------------------------------
// ST{ADD|CLR|SET)(A|L|AL)T instructions
//------------------------------------------------------------------------------

sttadd w0, [x2]
// CHECK-INST: sttadd w0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x04,0x20,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  1920045f <unknown>

sttadd w2, [sp]
// CHECK-INST: sttadd w2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0x22,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192207ff <unknown>

sttadd x0, [x2]
// CHECK-INST: sttadd x0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x04,0x20,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  5920045f <unknown>

sttadd x2, [sp]
// CHECK-INST: sttadd x2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0x22,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592207ff <unknown>

sttaddl w0, [x2]
// CHECK-INST: sttaddl w0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x04,0x60,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  1960045f <unknown>

sttaddl w2, [sp]
// CHECK-INST: sttaddl w2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0x62,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196207ff <unknown>

sttaddl x0, [x2]
// CHECK-INST: sttaddl x0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x04,0x60,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  5960045f <unknown>

sttaddl x2, [sp]
// CHECK-INST: sttaddl x2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x07,0x62,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596207ff <unknown>

sttclr w0, [x2]
// CHECK-INST: sttclr w0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x14,0x20,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  1920145f <unknown>

sttclr w2, [sp]
// CHECK-INST: sttclr w2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0x22,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192217ff <unknown>

sttclr x0, [x2]
// CHECK-INST: sttclr x0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x14,0x20,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  5920145f <unknown>

sttclr x2, [sp]
// CHECK-INST: sttclr x2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0x22,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592217ff <unknown>

sttclrl w0, [x2]
// CHECK-INST: sttclrl w0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x14,0x60,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  1960145f <unknown>

sttclrl w2, [sp]
// CHECK-INST: sttclrl w2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0x62,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196217ff <unknown>

sttclrl x0, [x2]
// CHECK-INST: sttclrl x0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x14,0x60,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  5960145f <unknown>

sttclrl x2, [sp]
// CHECK-INST: sttclrl x2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x17,0x62,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596217ff <unknown>

sttset w0, [x2]
// CHECK-INST: sttset w0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x34,0x20,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  1920345f <unknown>

sttset w2, [sp]
// CHECK-INST: sttset w2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0x22,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  192237ff <unknown>

sttset x0, [x2]
// CHECK-INST: sttset x0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x34,0x20,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  5920345f <unknown>

sttset x2, [sp]
// CHECK-INST: sttset x2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0x22,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  592237ff <unknown>

sttsetl w0, [x2]
// CHECK-INST: sttsetl w0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x34,0x60,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  1960345f <unknown>

sttsetl w2, [sp]
// CHECK-INST: sttsetl w2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0x62,0x19]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  196237ff <unknown>

sttsetl x0, [x2]
// CHECK-INST: sttsetl x0, [x2]
// CHECK-ENCODING: encoding: [0x5f,0x34,0x60,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  5960345f <unknown>

sttsetl x2, [sp]
// CHECK-INST: sttsetl x2, [sp]
// CHECK-ENCODING: encoding: [0xff,0x37,0x62,0x59]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  596237ff <unknown>

//------------------------------------------------------------------------------
// Load/store non-temporal register pair (offset)
//------------------------------------------------------------------------------
ldtnp x21, x29, [x2, #504]
// CHECK-INST: ldtnp x21, x29, [x2, #504]
// CHECK-ENCODING: encoding: [0x55,0xf4,0x5f,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e85ff455 <unknown>

ldtnp x22, x23, [x3, #-512]
// CHECK-INST: ldtnp x22, x23, [x3, #-512]
// CHECK-ENCODING: encoding: [0x76,0x5c,0x60,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e8605c76 <unknown>

ldtnp x24, x25, [x4, #8]
// CHECK-INST: ldtnp x24, x25, [x4, #8]
// CHECK-ENCODING: encoding: [0x98,0xe4,0x40,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e840e498 <unknown>

ldtnp q23, q29, [x1, #-1024]
// CHECK-INST: ldtnp q23, q29, [x1, #-1024]
// CHECK-ENCODING: encoding: [0x37,0x74,0x60,0xec]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ec607437 <unknown>

sttnp x3, x5, [sp]
// CHECK-INST: sttnp x3, x5, [sp]
// CHECK-ENCODING: encoding: [0xe3,0x17,0x00,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e80017e3 <unknown>

sttnp x17, x19, [sp, #64]
// CHECK-INST: sttnp x17, x19, [sp, #64]
// CHECK-ENCODING: encoding: [0xf1,0x4f,0x04,0xe8]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  e8044ff1 <unknown>

sttnp q3, q5, [sp]
// CHECK-INST: sttnp q3, q5, [sp]
// CHECK-ENCODING: encoding: [0xe3,0x17,0x00,0xec]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ec0017e3 <unknown>

sttnp q17, q19, [sp, #1008]
// CHECK-INST: sttnp q17, q19, [sp, #1008]
// CHECK-ENCODING: encoding: [0xf1,0xcf,0x1f,0xec]
// CHECK-ERROR: error: instruction requires: lsui
// CHECK-UNKNOWN:  ec1fcff1 <unknown>
