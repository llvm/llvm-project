// RUN: llvm-mc -triple aarch64 -mattr=+chk -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -mattr=+v8.9a -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -mattr=+v9.4a -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -mattr=+v8a -show-encoding %s | FileCheck %s --check-prefix=NO-CHK

// FEAT_CHK is mandatory from v8.0-a, but a clang user may not be using the LLVM
// integrated assembler, so we cannot just print `chkfeat x16` in all
// circumstances. Thankfully, we can always print `hint #40` when we cannot
// print `chkfeat x16`.
//
// So, in this case, we only print `chkfeat x16` from v8.9-a onwards, as an
// assembler that understands v8.9-a will understand `chkfeat x16`, and those
// that understand previous versions may not.

chkfeat x16
// CHECK: chkfeat x16                       // encoding: [0x1f,0x25,0x03,0xd5]
// NO-CHK: hint #40                              // encoding: [0x1f,0x25,0x03,0xd5]

hint #40
// CHECK: chkfeat x16                      // encoding: [0x1f,0x25,0x03,0xd5]
// NO-CHK: hint #40                             // encoding: [0x1f,0x25,0x03,0xd5]
