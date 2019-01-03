// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+tlb-rmi < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-V84
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a,-tlb-rmi < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-V84

// Outer shareable TLB maintenance instructions:
tlbi vmalle1os
tlbi vae1os, xzr
tlbi vae1os, x0
tlbi aside1os, x1
tlbi vaae1os, x2
tlbi vale1os, x3
tlbi vaale1os, x4
tlbi ipas2e1os, x5
tlbi ipas2le1os, x6
tlbi vae2os, x7
tlbi vale2os, x8
tlbi vmalls12e1os
tlbi vae3os, x9
tlbi vale3os, x10
tlbi alle2os
tlbi alle1os
tlbi alle3os

//CHECK:       tlbi  vmalle1os               // encoding: [0x1f,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  vae1os, xzr             // encoding: [0x3f,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  vae1os, x0              // encoding: [0x20,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  aside1os, x1            // encoding: [0x41,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  vaae1os, x2             // encoding: [0x62,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  vale1os, x3             // encoding: [0xa3,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  vaale1os, x4            // encoding: [0xe4,0x81,0x08,0xd5]
//CHECK-NEXT:  tlbi  ipas2e1os, x5           // encoding: [0x05,0x84,0x0c,0xd5]
//CHECK-NEXT:  tlbi  ipas2le1os, x6          // encoding: [0x86,0x84,0x0c,0xd5]
//CHECK-NEXT:  tlbi  vae2os, x7              // encoding: [0x27,0x81,0x0c,0xd5]
//CHECK-NEXT:  tlbi  vale2os, x8             // encoding: [0xa8,0x81,0x0c,0xd5]
//CHECK-NEXT:  tlbi  vmalls12e1os            // encoding: [0xdf,0x81,0x0c,0xd5]
//CHECK-NEXT:  tlbi  vae3os, x9              // encoding: [0x29,0x81,0x0e,0xd5]
//CHECK-NEXT:  tlbi  vale3os, x10            // encoding: [0xaa,0x81,0x0e,0xd5]
//CHECK-NEXT:  tlbi  alle2os                 // encoding: [0x1f,0x81,0x0c,0xd5]
//CHECK-NEXT:  tlbi  alle1os                 // encoding: [0x9f,0x81,0x0c,0xd5]
//CHECK-NEXT:  tlbi  alle3os                 // encoding: [0x1f,0x81,0x0e,0xd5]

tlbi vae1os, sp

//CHECK-ERROR:      error: invalid operand for instruction
//CHECK-ERROR-NEXT: tlbi vae1os, sp
//CHECK-ERROR-NEXT:              ^

//CHECK-NO-V84:      error: TLBI VMALLE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vmalle1os
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VAE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vae1os, xzr
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VAE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vae1os, x0
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI ASIDE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi aside1os, x1
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VAAE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vaae1os, x2
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VALE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vale1os, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VAALE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vaale1os, x4
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI IPAS2E1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ipas2e1os, x5
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI IPAS2LE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ipas2le1os, x6
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VAE2OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vae2os, x7
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VALE2OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vale2os, x8
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VMALLS12E1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vmalls12e1os
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VAE3OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vae3os, x9
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI VALE3OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi vale3os, x10
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI ALLE2OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi alle2os
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI ALLE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi alle1os
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI ALLE3OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi alle3os
//CHECK-NO-V84-NEXT:      ^

// TLB Range maintenance instructions:
tlbi rvae1, x3
tlbi rvaae1, x3
tlbi rvale1, x3
tlbi rvaale1, x3
tlbi rvae1is, x3
tlbi rvaae1is, x3
tlbi rvale1is, x3
tlbi rvaale1is, x3
tlbi rvae1os, x3
tlbi rvaae1os, x3
tlbi rvale1os, x3
tlbi rvaale1os, x3
tlbi ripas2e1is, x3
tlbi ripas2le1is, x3
tlbi ripas2e1, X3
tlbi ripas2le1, X3
tlbi ripas2e1os, X3
tlbi ripas2le1os, X3
tlbi rvae2, X3
tlbi rvale2, X3
tlbi rvae2is, X3
tlbi rvale2is, X3
tlbi rvae2os, X3
tlbi rvale2os, X3
tlbi rvae3, X3
tlbi rvale3, X3
tlbi rvae3is, X3
tlbi rvale3is, X3
tlbi rvae3os, X3
tlbi rvale3os, X3
tlbi rvale3os, XZR

//CHECK:       tlbi  rvae1, x3               // encoding: [0x23,0x86,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvaae1, x3              // encoding: [0x63,0x86,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvale1, x3              // encoding: [0xa3,0x86,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvaale1, x3             // encoding: [0xe3,0x86,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvae1is, x3             // encoding: [0x23,0x82,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvaae1is, x3            // encoding: [0x63,0x82,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvale1is, x3            // encoding: [0xa3,0x82,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvaale1is, x3           // encoding: [0xe3,0x82,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvae1os, x3             // encoding: [0x23,0x85,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvaae1os, x3            // encoding: [0x63,0x85,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvale1os, x3            // encoding: [0xa3,0x85,0x08,0xd5]
//CHECK-NEXT:  tlbi  rvaale1os, x3           // encoding: [0xe3,0x85,0x08,0xd5]
//CHECK-NEXT:  tlbi  ripas2e1is, x3          // encoding: [0x43,0x80,0x0c,0xd5]
//CHECK-NEXT:  tlbi  ripas2le1is, x3         // encoding: [0xc3,0x80,0x0c,0xd5]
//CHECK-NEXT:  tlbi  ripas2e1, x3            // encoding: [0x43,0x84,0x0c,0xd5]
//CHECK-NEXT:  tlbi  ripas2le1, x3           // encoding: [0xc3,0x84,0x0c,0xd5]
//CHECK-NEXT:  tlbi  ripas2e1os, x3          // encoding: [0x63,0x84,0x0c,0xd5]
//CHECK-NEXT:  tlbi  ripas2le1os, x3         // encoding: [0xe3,0x84,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvae2, x3               // encoding: [0x23,0x86,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvale2, x3              // encoding: [0xa3,0x86,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvae2is, x3             // encoding: [0x23,0x82,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvale2is, x3            // encoding: [0xa3,0x82,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvae2os, x3             // encoding: [0x23,0x85,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvale2os, x3            // encoding: [0xa3,0x85,0x0c,0xd5]
//CHECK-NEXT:  tlbi  rvae3, x3               // encoding: [0x23,0x86,0x0e,0xd5]
//CHECK-NEXT:  tlbi  rvale3, x3              // encoding: [0xa3,0x86,0x0e,0xd5]
//CHECK-NEXT:  tlbi  rvae3is, x3             // encoding: [0x23,0x82,0x0e,0xd5]
//CHECK-NEXT:  tlbi  rvale3is, x3            // encoding: [0xa3,0x82,0x0e,0xd5]
//CHECK-NEXT:  tlbi  rvae3os, x3             // encoding: [0x23,0x85,0x0e,0xd5]
//CHECK-NEXT:  tlbi  rvale3os, x3            // encoding: [0xa3,0x85,0x0e,0xd5]
//CHECK-NEXT:  tlbi  rvale3os, xzr           // encoding: [0xbf,0x85,0x0e,0xd5]

tlbi rvae1, sp

//CHECK-ERROR:      error: invalid operand for instruction
//CHECK-ERROR-NEXT: tlbi rvae1, sp
//CHECK-ERROR-NEXT:             ^

//CHECK-NO-V84:      error: TLBI RVAE1 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae1, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAAE1 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvaae1, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE1 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale1, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAALE1 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvaale1, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT:  error: TLBI RVAE1IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae1is, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAAE1IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvaae1is, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE1IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale1is, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAALE1IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvaale1is, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae1os, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAAE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvaae1os, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale1os, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAALE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvaale1os, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RIPAS2E1IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ripas2e1is, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RIPAS2LE1IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ripas2le1is, x3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RIPAS2E1 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ripas2e1, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RIPAS2LE1 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ripas2le1, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RIPAS2E1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ripas2e1os, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RIPAS2LE1OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi ripas2le1os, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE2 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae2, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE2 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale2, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE2IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae2is, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE2IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale2is, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE2OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae2os, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE2OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale2os, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE3 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae3, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE3 requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale3, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE3IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae3is, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE3IS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale3is, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVAE3OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvae3os, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE3OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale3os, X3
//CHECK-NO-V84-NEXT:      ^
//CHECK-NO-V84-NEXT: error: TLBI RVALE3OS requires tlb-rmi
//CHECK-NO-V84-NEXT: tlbi rvale3os, XZR
//CHECK-NO-V84-NEXT:      ^
