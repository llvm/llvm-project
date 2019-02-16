// RUN:     llvm-mc -triple armv8   -show-encoding -mattr=+sb < %s      | FileCheck %s
// RUN:     llvm-mc -triple armv8   -show-encoding -mattr=+v8.5a    < %s      | FileCheck %s
// RUN: not llvm-mc -triple armv8   -show-encoding -mattr=-sb < %s 2>&1 | FileCheck %s --check-prefix=NOSB
// RUN:     llvm-mc -triple thumbv8 -show-encoding -mattr=+sb < %s      | FileCheck %s --check-prefix=THUMB
// RUN:     llvm-mc -triple thumbv8 -show-encoding -mattr=+v8.5a    < %s      | FileCheck %s --check-prefix=THUMB
// RUN: not llvm-mc -triple thumbv8 -show-encoding -mattr=-sb < %s 2>&1 | FileCheck %s --check-prefix=NOSB

// Flag manipulation
sb

// CHECK: sb    @ encoding: [0x70,0xf0,0x7f,0xf5]
// THUMB: sb    @ encoding: [0xbf,0xf3,0x70,0x8f]

// NOSB: instruction requires: sb
// NOSB-NEXT: sb
