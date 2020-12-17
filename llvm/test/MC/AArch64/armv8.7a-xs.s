// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a,+xs < %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERR --check-prefix=CHECK-XS-ERR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.7a < %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERR --check-prefix=CHECK-XS-ERR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.4a < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERR --check-prefix=CHECK-NO-XS-ERR %s < %t

  dsb #16
  dsb #20
  dsb #24
  dsb #28
  dsb oshnxs
  dsb nshnxs
  dsb ishnxs
  dsb synxs
// CHECK: dsb oshnxs                   // encoding: [0x3f,0x32,0x03,0xd5]
// CHECK: dsb nshnxs                   // encoding: [0x3f,0x36,0x03,0xd5]
// CHECK: dsb ishnxs                   // encoding: [0x3f,0x3a,0x03,0xd5]
// CHECK: dsb synxs                    // encoding: [0x3f,0x3e,0x03,0xd5]
// CHECK: dsb oshnxs                   // encoding: [0x3f,0x32,0x03,0xd5]
// CHECK: dsb nshnxs                   // encoding: [0x3f,0x36,0x03,0xd5]
// CHECK: dsb ishnxs                   // encoding: [0x3f,0x3a,0x03,0xd5]
// CHECK: dsb synxs                    // encoding: [0x3f,0x3e,0x03,0xd5]
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs
// CHECK-NO-XS-ERR: [[@LINE-16]]:3: error: instruction requires: xs

  dsb #17
  dsb nshstnxs
// CHECK-ERR: [[@LINE-2]]:8: error: barrier operand out of range
// CHECK-ERR: [[@LINE-2]]:7: error: invalid barrier option name

  tlbi ipas2e1isnxs, x1
  tlbi ipas2le1isnxs, x1
  tlbi vmalle1isnxs
  tlbi alle2isnxs
  tlbi alle3isnxs
  tlbi vae1isnxs, x1
  tlbi vae2isnxs, x1
  tlbi vae3isnxs, x1
  tlbi aside1isnxs, x1
  tlbi vaae1isnxs, x1
  tlbi alle1isnxs
  tlbi vale1isnxs, x1
  tlbi vale2isnxs, x1
  tlbi vale3isnxs, x1
  tlbi vmalls12e1isnxs
  tlbi vaale1isnxs, x1
  tlbi ipas2e1nxs, x1
  tlbi ipas2le1nxs, x1
  tlbi vmalle1nxs
  tlbi alle2nxs
  tlbi alle3nxs
  tlbi vae1nxs, x1
  tlbi vae2nxs, x1
  tlbi vae3nxs, x1
  tlbi aside1nxs, x1
  tlbi vaae1nxs, x1
  tlbi alle1nxs
  tlbi vale1nxs, x1
  tlbi vale2nxs, x1
  tlbi vale3nxs, x1
  tlbi vmalls12e1nxs
  tlbi vaale1nxs, x1
// CHECK: tlbi ipas2e1isnxs, x1        // encoding: [0x21,0x90,0x0c,0xd5]
// CHECK: tlbi ipas2le1isnxs, x1       // encoding: [0xa1,0x90,0x0c,0xd5]
// CHECK: tlbi vmalle1isnxs            // encoding: [0x1f,0x93,0x08,0xd5]
// CHECK: tlbi alle2isnxs              // encoding: [0x1f,0x93,0x0c,0xd5]
// CHECK: tlbi alle3isnxs              // encoding: [0x1f,0x93,0x0e,0xd5]
// CHECK: tlbi vae1isnxs, x1           // encoding: [0x21,0x93,0x08,0xd5]
// CHECK: tlbi vae2isnxs, x1           // encoding: [0x21,0x93,0x0c,0xd5]
// CHECK: tlbi vae3isnxs, x1           // encoding: [0x21,0x93,0x0e,0xd5]
// CHECK: tlbi aside1isnxs, x1         // encoding: [0x41,0x93,0x08,0xd5]
// CHECK: tlbi vaae1isnxs, x1          // encoding: [0x61,0x93,0x08,0xd5]
// CHECK: tlbi alle1isnxs              // encoding: [0x9f,0x93,0x0c,0xd5]
// CHECK: tlbi vale1isnxs, x1          // encoding: [0xa1,0x93,0x08,0xd5]
// CHECK: tlbi vale2isnxs, x1          // encoding: [0xa1,0x93,0x0c,0xd5]
// CHECK: tlbi vale3isnxs, x1          // encoding: [0xa1,0x93,0x0e,0xd5]
// CHECK: tlbi vmalls12e1isnxs         // encoding: [0xdf,0x93,0x0c,0xd5]
// CHECK: tlbi vaale1isnxs, x1         // encoding: [0xe1,0x93,0x08,0xd5]
// CHECK: tlbi ipas2e1nxs, x1          // encoding: [0x21,0x94,0x0c,0xd5]
// CHECK: tlbi ipas2le1nxs, x1         // encoding: [0xa1,0x94,0x0c,0xd5]
// CHECK: tlbi vmalle1nxs              // encoding: [0x1f,0x97,0x08,0xd5]
// CHECK: tlbi alle2nxs                // encoding: [0x1f,0x97,0x0c,0xd5]
// CHECK: tlbi alle3nxs                // encoding: [0x1f,0x97,0x0e,0xd5]
// CHECK: tlbi vae1nxs, x1             // encoding: [0x21,0x97,0x08,0xd5]
// CHECK: tlbi vae2nxs, x1             // encoding: [0x21,0x97,0x0c,0xd5]
// CHECK: tlbi vae3nxs, x1             // encoding: [0x21,0x97,0x0e,0xd5]
// CHECK: tlbi aside1nxs, x1           // encoding: [0x41,0x97,0x08,0xd5]
// CHECK: tlbi vaae1nxs, x1            // encoding: [0x61,0x97,0x08,0xd5]
// CHECK: tlbi alle1nxs                // encoding: [0x9f,0x97,0x0c,0xd5]
// CHECK: tlbi vale1nxs, x1            // encoding: [0xa1,0x97,0x08,0xd5]
// CHECK: tlbi vale2nxs, x1            // encoding: [0xa1,0x97,0x0c,0xd5]
// CHECK: tlbi vale3nxs, x1            // encoding: [0xa1,0x97,0x0e,0xd5]
// CHECK: tlbi vmalls12e1nxs           // encoding: [0xdf,0x97,0x0c,0xd5]
// CHECK: tlbi vaale1nxs, x1           // encoding: [0xe1,0x97,0x08,0xd5]
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI IPAS2E1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI IPAS2LE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VMALLE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ALLE2ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ALLE3ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAE2ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAE3ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ASIDE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAAE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ALLE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE2ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE3ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VMALLS12E1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAALE1ISnXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI IPAS2E1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI IPAS2LE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VMALLE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ALLE2nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ALLE3nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAE2nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAE3nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ASIDE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAAE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI ALLE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE2nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE2nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VALE3nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VMALLS12E1nXS requires: xs
// CHECK_NO_XS_ERR: [[@LINE-64]]:8: error: TLBI VAALE1nXS requires: xs

  tlbi vmalle1osnxs
  tlbi vae1osnxs, x1
  tlbi aside1osnxs, x1
  tlbi vaae1osnxs, x1
  tlbi vale1osnxs, x1
  tlbi vaale1osnxs, x1
  tlbi ipas2e1osnxs, x1
  tlbi ipas2le1osnxs, x1
  tlbi vae2osnxs, x1
  tlbi vale2osnxs, x1
  tlbi vmalls12e1osnxs
  tlbi vae3osnxs, x1
  tlbi vale3osnxs, x1
  tlbi alle2osnxs
  tlbi alle1osnxs
  tlbi alle3osnxs
  tlbi rvae1nxs, x1
  tlbi rvaae1nxs, x1
  tlbi rvale1nxs, x1
  tlbi rvaale1nxs, x1
  tlbi rvae1isnxs, x1
  tlbi rvaae1isnxs, x1
  tlbi rvale1isnxs, x1
  tlbi rvaale1isnxs, x1
  tlbi rvae1osnxs, x1
  tlbi rvaae1osnxs, x1
  tlbi rvale1osnxs, x1
  tlbi rvaale1osnxs, x1
  tlbi ripas2e1isnxs, x1
  tlbi ripas2le1isnxs, x1
  tlbi ripas2e1nxs, x1
  tlbi ripas2le1nxs, x1
  tlbi ripas2e1osnxs, x1
  tlbi ripas2le1osnxs, x1
  tlbi rvae2nxs, x1
  tlbi rvale2nxs, x1
  tlbi rvae2isnxs, x1
  tlbi rvale2isnxs, x1
  tlbi rvae2osnxs, x1
  tlbi rvale2osnxs, x1
  tlbi rvae3nxs, x1
  tlbi rvale3nxs, x1
  tlbi rvae3isnxs, x1
  tlbi rvale3isnxs, x1
  tlbi rvae3osnxs, x1
  tlbi rvale3osnxs, x1
// CHECK: tlbi vmalle1osnxs            // encoding: [0x1f,0x91,0x08,0xd5]
// CHECK: tlbi vae1osnxs, x1           // encoding: [0x21,0x91,0x08,0xd5]
// CHECK: tlbi aside1osnxs, x1         // encoding: [0x41,0x91,0x08,0xd5]
// CHECK: tlbi vaae1osnxs, x1          // encoding: [0x61,0x91,0x08,0xd5]
// CHECK: tlbi vale1osnxs, x1          // encoding: [0xa1,0x91,0x08,0xd5]
// CHECK: tlbi vaale1osnxs, x1         // encoding: [0xe1,0x91,0x08,0xd5]
// CHECK: tlbi ipas2e1osnxs, x1        // encoding: [0x01,0x94,0x0c,0xd5]
// CHECK: tlbi ipas2le1osnxs, x1       // encoding: [0x81,0x94,0x0c,0xd5]
// CHECK: tlbi vae2osnxs, x1           // encoding: [0x21,0x91,0x0c,0xd5]
// CHECK: tlbi vale2osnxs, x1          // encoding: [0xa1,0x91,0x0c,0xd5]
// CHECK: tlbi vmalls12e1osnxs         // encoding: [0xdf,0x91,0x0c,0xd5]
// CHECK: tlbi vae3osnxs, x1           // encoding: [0x21,0x91,0x0e,0xd5]
// CHECK: tlbi vale3osnxs, x1          // encoding: [0xa1,0x91,0x0e,0xd5]
// CHECK: tlbi alle2osnxs              // encoding: [0x1f,0x91,0x0c,0xd5]
// CHECK: tlbi alle1osnxs              // encoding: [0x9f,0x91,0x0c,0xd5]
// CHECK: tlbi alle3osnxs              // encoding: [0x1f,0x91,0x0e,0xd5]
// CHECK: tlbi rvae1nxs, x1            // encoding: [0x21,0x96,0x08,0xd5]
// CHECK: tlbi rvaae1nxs, x1           // encoding: [0x61,0x96,0x08,0xd5]
// CHECK: tlbi rvale1nxs, x1           // encoding: [0xa1,0x96,0x08,0xd5]
// CHECK: tlbi rvaale1nxs, x1          // encoding: [0xe1,0x96,0x08,0xd5]
// CHECK: tlbi rvae1isnxs, x1          // encoding: [0x21,0x92,0x08,0xd5]
// CHECK: tlbi rvaae1isnxs, x1         // encoding: [0x61,0x92,0x08,0xd5]
// CHECK: tlbi rvale1isnxs, x1         // encoding: [0xa1,0x92,0x08,0xd5]
// CHECK: tlbi rvaale1isnxs, x1        // encoding: [0xe1,0x92,0x08,0xd5]
// CHECK: tlbi rvae1osnxs, x1          // encoding: [0x21,0x95,0x08,0xd5]
// CHECK: tlbi rvaae1osnxs, x1         // encoding: [0x61,0x95,0x08,0xd5]
// CHECK: tlbi rvale1osnxs, x1         // encoding: [0xa1,0x95,0x08,0xd5]
// CHECK: tlbi rvaale1osnxs, x1        // encoding: [0xe1,0x95,0x08,0xd5]
// CHECK: tlbi ripas2e1isnxs, x1       // encoding: [0x41,0x90,0x0c,0xd5]
// CHECK: tlbi ripas2le1isnxs, x1      // encoding: [0xc1,0x90,0x0c,0xd5]
// CHECK: tlbi ripas2e1nxs, x1         // encoding: [0x41,0x94,0x0c,0xd5]
// CHECK: tlbi ripas2le1nxs, x1        // encoding: [0xc1,0x94,0x0c,0xd5]
// CHECK: tlbi ripas2e1osnxs, x1       // encoding: [0x61,0x94,0x0c,0xd5]
// CHECK: tlbi ripas2le1osnxs, x1      // encoding: [0xe1,0x94,0x0c,0xd5]
// CHECK: tlbi rvae2nxs, x1            // encoding: [0x21,0x96,0x0c,0xd5]
// CHECK: tlbi rvale2nxs, x1           // encoding: [0xa1,0x96,0x0c,0xd5]
// CHECK: tlbi rvae2isnxs, x1          // encoding: [0x21,0x92,0x0c,0xd5]
// CHECK: tlbi rvale2isnxs, x1         // encoding: [0xa1,0x92,0x0c,0xd5]
// CHECK: tlbi rvae2osnxs, x1          // encoding: [0x21,0x95,0x0c,0xd5]
// CHECK: tlbi rvale2osnxs, x1         // encoding: [0xa1,0x95,0x0c,0xd5]
// CHECK: tlbi rvae3nxs, x1            // encoding: [0x21,0x96,0x0e,0xd5]
// CHECK: tlbi rvale3nxs, x1           // encoding: [0xa1,0x96,0x0e,0xd5]
// CHECK: tlbi rvae3isnxs, x1          // encoding: [0x21,0x92,0x0e,0xd5]
// CHECK: tlbi rvale3isnxs, x1         // encoding: [0xa1,0x92,0x0e,0xd5]
// CHECK: tlbi rvae3osnxs, x1          // encoding: [0x21,0x95,0x0e,0xd5]
// CHECK: tlbi rvale3osnxs, x1         // encoding: [0xa1,0x95,0x0e,0xd5]
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VMALLE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VAE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI ASIDE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VAAE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VALE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VAALE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI IPAS2E1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI IPAS2LE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VAE2OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VALE2OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VMALLS12E1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VAE3OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI VALE3OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI ALLE2OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI ALLE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI ALLE3OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE1nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAAE1nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE1nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAALE1nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE1ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAAE1ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE1ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAALE1ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAAE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAALE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RIPAS2E1ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RIPAS2LE1ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RIPAS2E1nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RIPAS2LE1nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RIPAS2E1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RIPAS2LE1OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE2nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE2nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE2ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE2ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE2OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE2OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE3nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE3nXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE3ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE3ISnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVAE3OSnXS requires: tlb-rmi, xs
// CHECK_NO_XS_ERR: [[@LINE-92]]:8: error: TLBI RVALE3OSnXS requires: tlb-rmi, xs
