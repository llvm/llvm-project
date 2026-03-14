// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a,+altnzcv < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a,-v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOALTFP
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOALTFP

// Flag manipulation
xaflag
axflag

// CHECK: xaflag                          // encoding: [0x3f,0x40,0x00,0xd5]
// CHECK: axflag                          // encoding: [0x5f,0x40,0x00,0xd5]

// NOALTFP: instruction requires: altnzcv
// NOALTFP-NEXT: xaflag
// NOALTFP: instruction requires: altnzcv
// NOALTFP-NEXT: axflag
