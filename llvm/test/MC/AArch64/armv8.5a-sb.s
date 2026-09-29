// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+sb -o - %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a -o - %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-sb -o - %s 2>&1 | FileCheck %s --check-prefix=NOSB

// Flag manipulation
sb

// CHECK: sb // encoding: [0xff,0x30,0x03,0xd5]

// NOSB: instruction requires: sb
// NOSB-NEXT: sb
