// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cmpbr < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmpbr < %s \
// RUN:        | llvm-objdump -d  --no-print-imm-hex --mattr=+cmpbr - | FileCheck %s --check-prefix=CHECK-DISASS
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmpbr < %s \
// RUN:        | llvm-objdump -d  --no-print-imm-hex --mattr=-cmpbr - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cmpbr < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+cmpbr -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// Compare & branch (Register)
//------------------------------------------------------------------------------

cbgt w5, w5, #-1024
// CHECK-INST: cbgt w5, w5, #-1024
// CHECK-DISASS: cbgt w5, w5,  0xfffffffffffffc00 <.text+0xfffffffffffffc00
// CHECK-ENCODING: [0x05,0x20,0x05,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74052005 <unknown>

cbgt x5, x5, #1020
// CHECK-INST: cbgt x5, x5, #1020
// CHECK-DISASS: cbgt x5, x5, 0x400 <.text+0x400>
// CHECK-ENCODING: [0xe5,0x1f,0x05,0xf4]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4051fe5 <unknown>

cbge x2, x2, #-1024
// CHECK-INST: cbge x2, x2, #-1024
// CHECK-DISASS: cbge x2, x2, 0xfffffffffffffc08 <.text+0xfffffffffffffc08
// CHECK-ENCODING: [0x02,0x20,0x22,0xf4]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4222002 <unknown>

cbge w5, w5, #1020
// CHECK-INST: cbge w5, w5, #1020
// CHECK-DISASS: cbge w5, w5, 0x408 <.text+0x408>
// CHECK-ENCODING: [0xe5,0x1f,0x25,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74251fe5 <unknown>


cbhi w5, w5, #-1024
// CHECK-INST: cbhi w5, w5, #-1024
// CHECK-DISASS: cbhi w5, w5, 0xfffffffffffffc10 <.text+0xfffffffffffffc10>
// CHECK-ENCODING: [0x05,0x20,0x45,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74452005 <unknown>

cbhi x5, x5, #1020
// CHECK-INST: cbhi x5, x5, #1020
// CHECK-DISASS: cbhi x5, x5, 0x410 <.text+0x410>
// CHECK-ENCODING: [0xe5,0x1f,0x45,0xf4]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4451fe5 <unknown>

cbhs x2, x2, #1020
// CHECK-INST: cbhs x2, x2, #1020
// CHECK-DISASS: cbhs x2, x2, 0x414 <.text+0x414>
// CHECK-ENCODING: [0xe2,0x1f,0x62,0xf4]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4621fe2 <unknown>

cbhs w5, w5, #1020
// CHECK-INST: cbhs w5, w5, #1020
// CHECK-DISASS: cbhs w5, w5, 0x418 <.text+0x418>
// CHECK-ENCODING: [0xe5,0x1f,0x65,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74651fe5 <unknown>

cbeq w5, w5, #-1024
// CHECK-INST: cbeq w5, w5, #-1024
// CHECK-DISASS: cbeq w5, w5, 0xfffffffffffffc20 <.text+0xfffffffffffffc20>
// CHECK-ENCODING: [0x05,0x20,0xc5,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74c52005 <unknown>

cbeq x5, x5, #1020
// CHECK-INST: cbeq x5, x5, #1020
// CHECK-DISASS: cbeq x5, x5, 0x420 <.text+0x420>
// CHECK-ENCODING: [0xe5,0x1f,0xc5,0xf4]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4c51fe5 <unknown>

cbne x2, x2, #-1024
// CHECK-INST: cbne x2, x2, #-1024
// CHECK-DISASS: cbne x2, x2, 0xfffffffffffffc28 <.text+0xfffffffffffffc28>
// CHECK-ENCODING: [0x02,0x20,0xe2,0xf4]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4e22002 <unknown>

cbne w5, w5, #-1024
// CHECK-INST: cbne w5, w5, #-1024
// CHECK-DISASS: cbne w5, w5, 0xfffffffffffffc2c <.text+0xfffffffffffffc2c>
// CHECK-ENCODING: [0x05,0x20,0xe5,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74e52005 <unknown>

///
// CBH<XX>
///

cbhgt w5, w5, #-1024
// CHECK-INST: cbhgt w5, w5, #-1024
// CHECK-DISASS: cbhgt w5, w5, 0xfffffffffffffc30 <.text+0xfffffffffffffc30>
// CHECK-ENCODING: [0x05,0xe0,0x05,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7405e005 <unknown>

cbhge w5, w5, #-1024
// CHECK-INST: cbhge w5, w5, #-1024
// CHECK-DISASS: cbhge w5, w5, 0xfffffffffffffc34 <.text+0xfffffffffffffc34>
// CHECK-ENCODING: [0x05,0xe0,0x25,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7425e005 <unknown>

cbhhi w5, w5, #-1024
// CHECK-INST: cbhhi w5, w5, #-1024
// CHECK-DISASS: cbhhi w5, w5, 0xfffffffffffffc38 <.text+0xfffffffffffffc38>
// CHECK-ENCODING: [0x05,0xe0,0x45,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7445e005 <unknown>

cbhhs w5, w5, #-1024
// CHECK-INST: cbhhs w5, w5, #-1024
// CHECK-DISASS: cbhhs w5, w5, 0xfffffffffffffc3c <.text+0xfffffffffffffc3c>
// CHECK-ENCODING: [0x05,0xe0,0x65,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7465e005 <unknown>

cbheq w5, w5, #-1024
// CHECK-INST: cbheq w5, w5, #-1024
// CHECK-DISASS: cbheq w5, w5, 0xfffffffffffffc40 <.text+0xfffffffffffffc40>
// CHECK-ENCODING: [0x05,0xe0,0xc5,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74c5e005 <unknown>

cbhne w5, w5, #-1024
// CHECK-INST: cbhne w5, w5, #-1024
// CHECK-DISASS: cbhne w5, w5, 0xfffffffffffffc44 <.text+0xfffffffffffffc44>
// CHECK-ENCODING: [0x05,0xe0,0xe5,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74e5e005 <unknown>


///
// CBB<XX>
///
cbbgt w5, w5, #-1024
// CHECK-INST: cbbgt w5, w5, #-1024
// CHECK-DISASS: cbbgt w5, w5, 0xfffffffffffffc48 <.text+0xfffffffffffffc48>
// CHECK-ENCODING: [0x05,0xa0,0x05,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7405a005 <unknown>

cbbge w5, w5, #-1024
// CHECK-INST: cbbge w5, w5, #-1024
// CHECK-DISASS: cbbge w5, w5, 0xfffffffffffffc4c <.text+0xfffffffffffffc4c>
// CHECK-ENCODING: [0x05,0xa0,0x25,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7425a005 <unknown>

cbbhi w5, w5, #-1024
// CHECK-INST: cbbhi w5, w5, #-1024
// CHECK-DISASS: cbbhi w5, w5, 0xfffffffffffffc50 <.text+0xfffffffffffffc50>
// CHECK-ENCODING: [0x05,0xa0,0x45,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7445a005 <unknown>

cbbhs w5, w5, #-1024
// CHECK-INST: cbbhs w5, w5, #-1024
// CHECK-DISASS: cbbhs w5, w5, 0xfffffffffffffc54 <.text+0xfffffffffffffc54>
// CHECK-ENCODING: [0x05,0xa0,0x65,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7465a005 <unknown>

cbbeq w5, w5, #-1024
// CHECK-INST: cbbeq w5, w5, #-1024
// CHECK-DISASS: cbbeq w5, w5, 0xfffffffffffffc58 <.text+0xfffffffffffffc58>
// CHECK-ENCODING: [0x05,0xa0,0xc5,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74c5a005 <unknown>

cbbne w5, w5, #-1024
// CHECK-INST: cbbne w5, w5, #-1024
// CHECK-DISASS: cbbne w5, w5, 0xfffffffffffffc5c <.text+0xfffffffffffffc5c>
// CHECK-ENCODING: [0x05,0xa0,0xe5,0x74]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74e5a005 <unknown>

//------------------------------------------------------------------------------
// Compare & branch (Immediate)
//------------------------------------------------------------------------------

cbgt w5, #63, #1020
// CHECK-INST: cbgt w5, #63, #1020
// CHECK-DISASS: cbgt w5,  #63, 0x45c <.text+0x45c>
// CHECK-ENCODING: [0xe5,0x9f,0x1f,0x75]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 751f9fe5 <unknown>

cbgt x5, #0, #-1024
// CHECK-INST: cbgt x5, #0, #-1024
// CHECK-DISASS: cbgt x5, #0, 0xfffffffffffffc64 <.text+0xfffffffffffffc64>
// CHECK-ENCODING: [0x05,0x20,0x00,0xf5]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5002005 <unknown>

cbhi w5, #31, #1020
// CHECK-INST: cbhi w5, #31, #1020
// CHECK-DISASS: cbhi w5, #31, 0x464 <.text+0x464>
// CHECK-ENCODING: [0xe5,0x9f,0x4f,0x75]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 754f9fe5 <unknown>

cbhi x5, #63, #-1024
// CHECK-INST: cbhi x5, #63, #-1024
// CHECK-DISASS: cbhi x5, #63, 0xfffffffffffffc6c <.text+0xfffffffffffffc6c>
// CHECK-ENCODING: [0x05,0xa0,0x5f,0xf5]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f55fa005 <unknown>

cblt w5, #63, #1020
// CHECK-INST: cblt w5, #63, #1020
// CHECK-DISASS: cblt w5, #63, 0x46c <.text+0x46c>
// CHECK-ENCODING: [0xe5,0x9f,0x3f,0x75]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 753f9fe5 <unknown>

cblt x5, #0, #-1024
// CHECK-INST: cblt x5, #0, #-1024
// CHECK-DISASS: cblt x5, #0, 0xfffffffffffffc74 <.text+0xfffffffffffffc74>
// CHECK-ENCODING: [0x05,0x20,0x20,0xf5]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5202005 <unknown>

cblo w5, #31, #1020
// CHECK-INST: cblo w5, #31, #1020
// CHECK-DISASS: cblo w5, #31, 0x474 <.text+0x474>
// CHECK-ENCODING: [0xe5,0x9f,0x6f,0x75]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 756f9fe5 <unknown>

cblo x5, #31, #-1024
// CHECK-INST: cblo x5, #31, #-1024
// CHECK-DISASS: cblo x5, #31, 0xfffffffffffffc7c <.text+0xfffffffffffffc7c>
// CHECK-ENCODING: [0x05,0xa0,0x6f,0xf5]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f56fa005 <unknown>

cbeq w5, #31, #1020
// CHECK-INST: cbeq w5, #31, #1020
// CHECK-DISASS: cbeq w5, #31, 0x47c <.text+0x47c>
// CHECK-ENCODING: [0xe5,0x9f,0xcf,0x75]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 75cf9fe5 <unknown>

cbeq x5, #31, #-1024
// CHECK-INST: cbeq x5, #31, #-1024
// CHECK-DISASS: cbeq x5, #31, 0xfffffffffffffc84 <.text+0xfffffffffffffc84>
// CHECK-ENCODING: [0x05,0xa0,0xcf,0xf5]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5cfa005 <unknown>

cbne w5, #31, #1020
// CHECK-INST: cbne w5, #31, #1020
// CHECK-DISASS: cbne w5, #31, 0x484 <.text+0x484>
// CHECK-ENCODING: [0xe5,0x9f,0xef,0x75]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 75ef9fe5 <unknown>

cbne x5, #31, #-1024
// CHECK-INST: cbne x5, #31, #-1024
// CHECK-DISASS: cbne x5, #31, 0xfffffffffffffc8c <.text+0xfffffffffffffc8c>
// CHECK-ENCODING: [0x05,0xa0,0xef,0xf5]
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5efa005 <unknown>