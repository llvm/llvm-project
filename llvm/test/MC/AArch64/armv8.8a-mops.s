// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+mops,+mte < %s 2> %t | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MTE
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a,+mte < %s 2> %t | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MTE
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+mops < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-NO-MTE-ERR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a < %s 2> %t | FileCheck %s --check-prefix=CHECK
// RUN: FileCheck --check-prefix=CHECK-NO-MTE-ERR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-MOPS-ERR --check-prefix=CHECK-NO-MOPSMTE-ERR %s < %t

// CHECK:      [0x40,0x04,0x01,0x19]
// CHECK-NEXT: [0x40,0x44,0x01,0x19]
// CHECK-NEXT: [0x40,0x84,0x01,0x19]
// CHECK-NEXT: [0x40,0xc4,0x01,0x19]
// CHECK-NEXT: [0x40,0x14,0x01,0x19]
// CHECK-NEXT: [0x40,0x54,0x01,0x19]
// CHECK-NEXT: [0x40,0x94,0x01,0x19]
// CHECK-NEXT: [0x40,0xd4,0x01,0x19]
// CHECK-NEXT: [0x40,0x24,0x01,0x19]
// CHECK-NEXT: [0x40,0x64,0x01,0x19]
// CHECK-NEXT: [0x40,0xa4,0x01,0x19]
// CHECK-NEXT: [0x40,0xe4,0x01,0x19]
// CHECK-NEXT: [0x40,0x34,0x01,0x19]
// CHECK-NEXT: [0x40,0x74,0x01,0x19]
// CHECK-NEXT: [0x40,0xb4,0x01,0x19]
// CHECK-NEXT: [0x40,0xf4,0x01,0x19]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
cpyfp [x0]!, [x1]!, x2!
cpyfpwn [x0]!, [x1]!, x2!
cpyfprn [x0]!, [x1]!, x2!
cpyfpn [x0]!, [x1]!, x2!
cpyfpwt [x0]!, [x1]!, x2!
cpyfpwtwn [x0]!, [x1]!, x2!
cpyfpwtrn [x0]!, [x1]!, x2!
cpyfpwtn [x0]!, [x1]!, x2!
cpyfprt [x0]!, [x1]!, x2!
cpyfprtwn [x0]!, [x1]!, x2!
cpyfprtrn [x0]!, [x1]!, x2!
cpyfprtn [x0]!, [x1]!, x2!
cpyfpt [x0]!, [x1]!, x2!
cpyfptwn [x0]!, [x1]!, x2!
cpyfptrn [x0]!, [x1]!, x2!
cpyfptn [x0]!, [x1]!, x2!

// CHECK:      [0x40,0x04,0x41,0x19]
// CHECK-NEXT: [0x40,0x44,0x41,0x19]
// CHECK-NEXT: [0x40,0x84,0x41,0x19]
// CHECK-NEXT: [0x40,0xc4,0x41,0x19]
// CHECK-NEXT: [0x40,0x14,0x41,0x19]
// CHECK-NEXT: [0x40,0x54,0x41,0x19]
// CHECK-NEXT: [0x40,0x94,0x41,0x19]
// CHECK-NEXT: [0x40,0xd4,0x41,0x19]
// CHECK-NEXT: [0x40,0x24,0x41,0x19]
// CHECK-NEXT: [0x40,0x64,0x41,0x19]
// CHECK-NEXT: [0x40,0xa4,0x41,0x19]
// CHECK-NEXT: [0x40,0xe4,0x41,0x19]
// CHECK-NEXT: [0x40,0x34,0x41,0x19]
// CHECK-NEXT: [0x40,0x74,0x41,0x19]
// CHECK-NEXT: [0x40,0xb4,0x41,0x19]
// CHECK-NEXT: [0x40,0xf4,0x41,0x19]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
cpyfm [x0]!, [x1]!, x2!
cpyfmwn [x0]!, [x1]!, x2!
cpyfmrn [x0]!, [x1]!, x2!
cpyfmn [x0]!, [x1]!, x2!
cpyfmwt [x0]!, [x1]!, x2!
cpyfmwtwn [x0]!, [x1]!, x2!
cpyfmwtrn [x0]!, [x1]!, x2!
cpyfmwtn [x0]!, [x1]!, x2!
cpyfmrt [x0]!, [x1]!, x2!
cpyfmrtwn [x0]!, [x1]!, x2!
cpyfmrtrn [x0]!, [x1]!, x2!
cpyfmrtn [x0]!, [x1]!, x2!
cpyfmt [x0]!, [x1]!, x2!
cpyfmtwn [x0]!, [x1]!, x2!
cpyfmtrn [x0]!, [x1]!, x2!
cpyfmtn [x0]!, [x1]!, x2!

// CHECK:      [0x40,0x04,0x81,0x19]
// CHECK-NEXT: [0x40,0x44,0x81,0x19]
// CHECK-NEXT: [0x40,0x84,0x81,0x19]
// CHECK-NEXT: [0x40,0xc4,0x81,0x19]
// CHECK-NEXT: [0x40,0x14,0x81,0x19]
// CHECK-NEXT: [0x40,0x54,0x81,0x19]
// CHECK-NEXT: [0x40,0x94,0x81,0x19]
// CHECK-NEXT: [0x40,0xd4,0x81,0x19]
// CHECK-NEXT: [0x40,0x24,0x81,0x19]
// CHECK-NEXT: [0x40,0x64,0x81,0x19]
// CHECK-NEXT: [0x40,0xa4,0x81,0x19]
// CHECK-NEXT: [0x40,0xe4,0x81,0x19]
// CHECK-NEXT: [0x40,0x34,0x81,0x19]
// CHECK-NEXT: [0x40,0x74,0x81,0x19]
// CHECK-NEXT: [0x40,0xb4,0x81,0x19]
// CHECK-NEXT: [0x40,0xf4,0x81,0x19]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
cpyfe [x0]!, [x1]!, x2!
cpyfewn [x0]!, [x1]!, x2!
cpyfern [x0]!, [x1]!, x2!
cpyfen [x0]!, [x1]!, x2!
cpyfewt [x0]!, [x1]!, x2!
cpyfewtwn [x0]!, [x1]!, x2!
cpyfewtrn [x0]!, [x1]!, x2!
cpyfewtn [x0]!, [x1]!, x2!
cpyfert [x0]!, [x1]!, x2!
cpyfertwn [x0]!, [x1]!, x2!
cpyfertrn [x0]!, [x1]!, x2!
cpyfertn [x0]!, [x1]!, x2!
cpyfet [x0]!, [x1]!, x2!
cpyfetwn [x0]!, [x1]!, x2!
cpyfetrn [x0]!, [x1]!, x2!
cpyfetn [x0]!, [x1]!, x2!

// CHECK:      [0x40,0x04,0x01,0x1d]
// CHECK-NEXT: [0x40,0x44,0x01,0x1d]
// CHECK-NEXT: [0x40,0x84,0x01,0x1d]
// CHECK-NEXT: [0x40,0xc4,0x01,0x1d]
// CHECK-NEXT: [0x40,0x14,0x01,0x1d]
// CHECK-NEXT: [0x40,0x54,0x01,0x1d]
// CHECK-NEXT: [0x40,0x94,0x01,0x1d]
// CHECK-NEXT: [0x40,0xd4,0x01,0x1d]
// CHECK-NEXT: [0x40,0x24,0x01,0x1d]
// CHECK-NEXT: [0x40,0x64,0x01,0x1d]
// CHECK-NEXT: [0x40,0xa4,0x01,0x1d]
// CHECK-NEXT: [0x40,0xe4,0x01,0x1d]
// CHECK-NEXT: [0x40,0x34,0x01,0x1d]
// CHECK-NEXT: [0x40,0x74,0x01,0x1d]
// CHECK-NEXT: [0x40,0xb4,0x01,0x1d]
// CHECK-NEXT: [0x40,0xf4,0x01,0x1d]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
cpyp [x0]!, [x1]!, x2!
cpypwn [x0]!, [x1]!, x2!
cpyprn [x0]!, [x1]!, x2!
cpypn [x0]!, [x1]!, x2!
cpypwt [x0]!, [x1]!, x2!
cpypwtwn [x0]!, [x1]!, x2!
cpypwtrn [x0]!, [x1]!, x2!
cpypwtn [x0]!, [x1]!, x2!
cpyprt [x0]!, [x1]!, x2!
cpyprtwn [x0]!, [x1]!, x2!
cpyprtrn [x0]!, [x1]!, x2!
cpyprtn [x0]!, [x1]!, x2!
cpypt [x0]!, [x1]!, x2!
cpyptwn [x0]!, [x1]!, x2!
cpyptrn [x0]!, [x1]!, x2!
cpyptn [x0]!, [x1]!, x2!

// CHECK:      [0x40,0x04,0x41,0x1d]
// CHECK-NEXT: [0x40,0x44,0x41,0x1d]
// CHECK-NEXT: [0x40,0x84,0x41,0x1d]
// CHECK-NEXT: [0x40,0xc4,0x41,0x1d]
// CHECK-NEXT: [0x40,0x14,0x41,0x1d]
// CHECK-NEXT: [0x40,0x54,0x41,0x1d]
// CHECK-NEXT: [0x40,0x94,0x41,0x1d]
// CHECK-NEXT: [0x40,0xd4,0x41,0x1d]
// CHECK-NEXT: [0x40,0x24,0x41,0x1d]
// CHECK-NEXT: [0x40,0x64,0x41,0x1d]
// CHECK-NEXT: [0x40,0xa4,0x41,0x1d]
// CHECK-NEXT: [0x40,0xe4,0x41,0x1d]
// CHECK-NEXT: [0x40,0x34,0x41,0x1d]
// CHECK-NEXT: [0x40,0x74,0x41,0x1d]
// CHECK-NEXT: [0x40,0xb4,0x41,0x1d]
// CHECK-NEXT: [0x40,0xf4,0x41,0x1d]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
cpym [x0]!, [x1]!, x2!
cpymwn [x0]!, [x1]!, x2!
cpymrn [x0]!, [x1]!, x2!
cpymn [x0]!, [x1]!, x2!
cpymwt [x0]!, [x1]!, x2!
cpymwtwn [x0]!, [x1]!, x2!
cpymwtrn [x0]!, [x1]!, x2!
cpymwtn [x0]!, [x1]!, x2!
cpymrt [x0]!, [x1]!, x2!
cpymrtwn [x0]!, [x1]!, x2!
cpymrtrn [x0]!, [x1]!, x2!
cpymrtn [x0]!, [x1]!, x2!
cpymt [x0]!, [x1]!, x2!
cpymtwn [x0]!, [x1]!, x2!
cpymtrn [x0]!, [x1]!, x2!
cpymtn [x0]!, [x1]!, x2!

// CHECK:      [0x40,0x04,0x81,0x1d]
// CHECK-NEXT: [0x40,0x44,0x81,0x1d]
// CHECK-NEXT: [0x40,0x84,0x81,0x1d]
// CHECK-NEXT: [0x40,0xc4,0x81,0x1d]
// CHECK-NEXT: [0x40,0x14,0x81,0x1d]
// CHECK-NEXT: [0x40,0x54,0x81,0x1d]
// CHECK-NEXT: [0x40,0x94,0x81,0x1d]
// CHECK-NEXT: [0x40,0xd4,0x81,0x1d]
// CHECK-NEXT: [0x40,0x24,0x81,0x1d]
// CHECK-NEXT: [0x40,0x64,0x81,0x1d]
// CHECK-NEXT: [0x40,0xa4,0x81,0x1d]
// CHECK-NEXT: [0x40,0xe4,0x81,0x1d]
// CHECK-NEXT: [0x40,0x34,0x81,0x1d]
// CHECK-NEXT: [0x40,0x74,0x81,0x1d]
// CHECK-NEXT: [0x40,0xb4,0x81,0x1d]
// CHECK-NEXT: [0x40,0xf4,0x81,0x1d]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
cpye [x0]!, [x1]!, x2!
cpyewn [x0]!, [x1]!, x2!
cpyern [x0]!, [x1]!, x2!
cpyen [x0]!, [x1]!, x2!
cpyewt [x0]!, [x1]!, x2!
cpyewtwn [x0]!, [x1]!, x2!
cpyewtrn [x0]!, [x1]!, x2!
cpyewtn [x0]!, [x1]!, x2!
cpyert [x0]!, [x1]!, x2!
cpyertwn [x0]!, [x1]!, x2!
cpyertrn [x0]!, [x1]!, x2!
cpyertn [x0]!, [x1]!, x2!
cpyet [x0]!, [x1]!, x2!
cpyetwn [x0]!, [x1]!, x2!
cpyetrn [x0]!, [x1]!, x2!
cpyetn [x0]!, [x1]!, x2!

// CHECK:      [0x20,0x04,0xc2,0x19]
// CHECK-NEXT: [0x20,0x14,0xc2,0x19]
// CHECK-NEXT: [0x20,0x24,0xc2,0x19]
// CHECK-NEXT: [0x20,0x34,0xc2,0x19]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
setp [x0]!, x1!, x2
setpt [x0]!, x1!, x2
setpn [x0]!, x1!, x2
setptn [x0]!, x1!, x2

// CHECK: [0x20,0x44,0xc2,0x19]
// CHECK: [0x20,0x54,0xc2,0x19]
// CHECK: [0x20,0x64,0xc2,0x19]
// CHECK: [0x20,0x74,0xc2,0x19]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
setm [x0]!, x1!, x2
setmt [x0]!, x1!, x2
setmn [x0]!, x1!, x2
setmtn [x0]!, x1!, x2

// CHECK: [0x20,0x84,0xc2,0x19]
// CHECK: [0x20,0x94,0xc2,0x19]
// CHECK: [0x20,0xa4,0xc2,0x19]
// CHECK: [0x20,0xb4,0xc2,0x19]
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
// CHECK-NO-MOPS-ERR: error: instruction requires: mops
sete [x0]!, x1!, x2
setet [x0]!, x1!, x2
seten [x0]!, x1!, x2
setetn [x0]!, x1!, x2

// CHECK-MTE: [0x20,0x04,0xc2,0x1d]
// CHECK-MTE: [0x20,0x14,0xc2,0x1d]
// CHECK-MTE: [0x20,0x24,0xc2,0x1d]
// CHECK-MTE: [0x20,0x34,0xc2,0x1d]
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
setgp [x0]!, x1!, x2
setgpt [x0]!, x1!, x2
setgpn [x0]!, x1!, x2
setgptn [x0]!, x1!, x2

// CHECK-MTE: [0x20,0x44,0xc2,0x1d]
// CHECK-MTE: [0x20,0x54,0xc2,0x1d]
// CHECK-MTE: [0x20,0x64,0xc2,0x1d]
// CHECK-MTE: [0x20,0x74,0xc2,0x1d]
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
setgm [x0]!, x1!, x2
setgmt [x0]!, x1!, x2
setgmn [x0]!, x1!, x2
setgmtn [x0]!, x1!, x2

// CHECK-MTE: [0x20,0x84,0xc2,0x1d]
// CHECK-MTE: [0x20,0x94,0xc2,0x1d]
// CHECK-MTE: [0x20,0xa4,0xc2,0x1d]
// CHECK-MTE: [0x20,0xb4,0xc2,0x1d]
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MTE-ERR: error: instruction requires: mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
// CHECK-NO-MOPSMTE-ERR: error: instruction requires: mops mte
setge [x0]!, x1!, x2
setget [x0]!, x1!, x2
setgen [x0]!, x1!, x2
setgetn [x0]!, x1!, x2

// All operand must be different from each other

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyfp [x0]!, [x0]!, x1!
cpyfp [x0]!, [x1]!, x0!
cpyfp [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyfm [x0]!, [x0]!, x1!
cpyfm [x0]!, [x1]!, x0!
cpyfm [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyfe [x0]!, [x0]!, x1!
cpyfe [x0]!, [x1]!, x0!
cpyfe [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyp [x0]!, [x0]!, x1!
cpyp [x0]!, [x1]!, x0!
cpyp [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpym [x0]!, [x0]!, x1!
cpym [x0]!, [x1]!, x0!
cpym [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpye [x0]!, [x0]!, x1!
cpye [x0]!, [x1]!, x0!
cpye [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setp [x0]!, x0!, x1
setp [x0]!, x1!, x0
setp [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setm [x0]!, x0!, x1
setm [x0]!, x1!, x0
setm [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
sete [x0]!, x0!, x1
sete [x0]!, x1!, x0
sete [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgp [x0]!, x0!, x1
setgp [x0]!, x1!, x0
setgp [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgm [x0]!, x0!, x1
setgm [x0]!, x1!, x0
setgm [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setge [x0]!, x0!, x1
setge [x0]!, x1!, x0
setge [x1]!, x0!, x0

// SP cannot be used as argument at any position

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfp [sp]!, [x1]!, x2!
cpyfp [x0]!, [sp]!, x2!
cpyfp [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfm [sp]!, [x1]!, x2!
cpyfm [x0]!, [sp]!, x2!
cpyfm [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfe [sp]!, [x1]!, x2!
cpyfe [x0]!, [sp]!, x2!
cpyfe [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyp [sp]!, [x2]!, x2!
cpyp [x0]!, [sp]!, x2!
cpyp [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpym [sp]!, [x2]!, x2!
cpym [x0]!, [sp]!, x2!
cpym [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpye [sp]!, [x2]!, x2!
cpye [x0]!, [sp]!, x2!
cpye [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setp [sp]!, x1!, x2
setp [x0]!, sp!, x2
setp [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setm [sp]!, x1!, x2
setm [x0]!, sp!, x2
setm [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
sete [sp]!, x1!, x2
sete [x0]!, sp!, x2
sete [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgp [sp]!, x1!, x2
setgp [x0]!, sp!, x2
setgp [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgm [sp]!, x1!, x2
setgm [x0]!, sp!, x2
setgm [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setge [sp]!, x1!, x2
setge [x0]!, sp!, x2
setge [x0]!, x1!, sp

// XZR can only be used at:
//  - the size operand in CPY.
//  - the size or source operands in SET.

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       cpyfp [x0]!, [x1]!, xzr!
cpyfp [xzr]!, [x1]!, x2!
cpyfp [x0]!, [xzr]!, x2!
cpyfp [x0]!, [x1]!, xzr!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       cpyfm [x0]!, [x1]!, xzr!
cpyfm [xzr]!, [x1]!, x2!
cpyfm [x0]!, [xzr]!, x2!
cpyfm [x0]!, [x1]!, xzr!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       cpyfe [x0]!, [x1]!, xzr!
cpyfe [xzr]!, [x1]!, x2!
cpyfe [x0]!, [xzr]!, x2!
cpyfe [x0]!, [x1]!, xzr!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       cpyp [x0]!, [x1]!, xzr!
cpyp [xzr]!, [x2]!, x2!
cpyp [x0]!, [xzr]!, x2!
cpyp [x0]!, [x1]!, xzr!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       cpym [x0]!, [x1]!, xzr!
cpym [xzr]!, [x2]!, x2!
cpym [x0]!, [xzr]!, x2!
cpym [x0]!, [x1]!, xzr!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       cpye [x0]!, [x1]!, xzr!
cpye [xzr]!, [x2]!, x2!
cpye [x0]!, [xzr]!, x2!
cpye [x0]!, [x1]!, xzr!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       setp [x0]!, xzr!, x2
// CHECK:       setp [x0]!, x1!, xzr
setp [xzr]!, x1!, x2
setp [x0]!, xzr!, x2
setp [x0]!, x1!, xzr

// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       setm [x0]!, xzr!, x2
// CHECK:       setm [x0]!, x1!, xzr
setm [xzr]!, x1!, x2
setm [x0]!, xzr!, x2
setm [x0]!, x1!, xzr

// CHECK-ERROR: error: invalid operand for instruction
// CHECK:       sete [x0]!, xzr!, x2
// CHECK:       sete [x0]!, x1!, xzr
sete [xzr]!, x1!, x2
sete [x0]!, xzr!, x2
sete [x0]!, x1!, xzr

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-MTE:   setgp [x0]!, xzr!, x2
// CHECK-MTE:   setgp [x0]!, x1!, xzr
setgp [xzr]!, x1!, x2
setgp [x0]!, xzr!, x2
setgp [x0]!, x1!, xzr

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-MTE:   setgm [x0]!, xzr!, x2
// CHECK-MTE:   setgm [x0]!, x1!, xzr
setgm [xzr]!, x1!, x2
setgm [x0]!, xzr!, x2
setgm [x0]!, x1!, xzr

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-MTE:   setge [x0]!, xzr!, x2
// CHECK-MTE:   setge [x0]!, x1!, xzr
setge [xzr]!, x1!, x2
setge [x0]!, xzr!, x2
setge [x0]!, x1!, xzr
