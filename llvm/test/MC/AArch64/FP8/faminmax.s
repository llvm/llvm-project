// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+faminmax < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+faminmax < %s \
// RUN:        | llvm-objdump -d --mattr=+faminmax - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+faminmax < %s \
// RUN:        | llvm-objdump -d --mattr=-faminmax - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+faminmax < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+faminmax -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

/// FAMAX instructions.
famax  v31.4h, v31.4h, v31.4h
// CHECK-INST: famax  v31.4h, v31.4h, v31.4h
// CHECK-ENCODING: [0xff,0x1f,0xdf,0x0e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 0edf1fff  <unknown>

famax  v31.4h, v0.4h, v31.4h
// CHECK-INST: famax  v31.4h, v0.4h, v31.4h
// CHECK-ENCODING: [0x1f,0x1c,0xdf,0x0e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 0edf1c1f <unknown>

famax  v0.4h, v0.4h, v0.4h
// CHECK-INST: famax  v0.4h, v0.4h, v0.4h
// CHECK-ENCODING: [0x00,0x1c,0xc0,0x0e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 0ec01c00 <unknown>

famax  v31.8h, v31.8h, v31.8h
// CHECK-INST: famax  v31.8h, v31.8h, v31.8h
// CHECK-ENCODING: [0xff,0x1f,0xdf,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4edf1fff <unknown>

famax  v31.8h, v31.8h, v0.8h
// CHECK-INST: famax  v31.8h, v31.8h, v0.8h
// CHECK-ENCODING: [0xff,0x1f,0xc0,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4ec01fff <unknown>

famax  v0.8h, v0.8h, v0.8h
// CHECK-INST: famax  v0.8h, v0.8h, v0.8h
// CHECK-ENCODING: [0x00,0x1c,0xc0,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4ec01c00 <unknown>

famax  v31.2s, v31.2s, v31.2s
// CHECK-INST: famax  v31.2s, v31.2s, v31.2s
// CHECK-ENCODING: [0xff,0xdf,0xbf,0x0e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 0ebfdfff <unknown>

famax  v31.2s, v0.2s, v0.2s
// CHECK-INST: famax  v31.2s, v0.2s, v0.2s
// CHECK-ENCODING: [0x1f,0xdc,0xa0,0x0e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 0ea0dc1f <unknown>

famax  v0.2s, v0.2s, v0.2s
// CHECK-INST: famax  v0.2s, v0.2s, v0.2s
// CHECK-ENCODING: [0x00,0xdc,0xa0,0x0e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 0ea0dc00 <unknown>

famax  v31.4s, v31.4s, v31.4s
// CHECK-INST: famax  v31.4s, v31.4s, v31.4s
// CHECK-ENCODING: [0xff,0xdf,0xbf,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4ebfdfff <unknown>

famax  v0.4s, v31.4s, v31.4s
// CHECK-INST: famax  v0.4s, v31.4s, v31.4s
// CHECK-ENCODING: [0xe0,0xdf,0xbf,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4ebfdfe0 <unknown>

famax  v0.4s, v0.4s, v0.4s
// CHECK-INST: famax  v0.4s, v0.4s, v0.4s
// CHECK-ENCODING: [0x00,0xdc,0xa0,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4ea0dc00 <unknown>

famax  v31.2d, v31.2d, v31.2d
// CHECK-INST: famax  v31.2d, v31.2d, v31.2d
// CHECK-ENCODING: [0xff,0xdf,0xff,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4effdfff <unknown>

famax  v0.2d, v0.2d, v31.2d
// CHECK-INST: famax  v0.2d, v0.2d, v31.2d
// CHECK-ENCODING: [0x00,0xdc,0xff,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4effdc00 <unknown>

famax  v0.2d, v0.2d, v0.2d
// CHECK-INST: famax  v0.2d, v0.2d, v0.2d
// CHECK-ENCODING: [0x00,0xdc,0xe0,0x4e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 4ee0dc00 <unknown>


/// FAMIN instructions.
famin  v31.4h, v31.4h, v31.4h
// CHECK-INST: famin  v31.4h, v31.4h, v31.4h
// CHECK-ENCODING: [0xff,0x1f,0xdf,0x2e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 2edf1fff  <unknown>

famin  v31.4h, v0.4h, v31.4h
// CHECK-INST: famin  v31.4h, v0.4h, v31.4h
// CHECK-ENCODING: [0x1f,0x1c,0xdf,0x2e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 2edf1c1f <unknown>

famin  v0.4h, v0.4h, v0.4h
// CHECK-INST: famin  v0.4h, v0.4h, v0.4h
// CHECK-ENCODING: [0x00,0x1c,0xc0,0x2e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 2ec01c00 <unknown>

famin  v31.8h, v31.8h, v31.8h
// CHECK-INST: famin  v31.8h, v31.8h, v31.8h
// CHECK-ENCODING: [0xff,0x1f,0xdf,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6edf1fff <unknown>

famin  v31.8h, v31.8h, v0.8h
// CHECK-INST: famin  v31.8h, v31.8h, v0.8h
// CHECK-ENCODING: [0xff,0x1f,0xc0,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6ec01fff <unknown>

famin  v0.8h, v0.8h, v0.8h
// CHECK-INST: famin  v0.8h, v0.8h, v0.8h
// CHECK-ENCODING: [0x00,0x1c,0xc0,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6ec01c00 <unknown>

famin  v31.2s, v31.2s, v31.2s
// CHECK-INST: famin  v31.2s, v31.2s, v31.2s
// CHECK-ENCODING: [0xff,0xdf,0xbf,0x2e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 2ebfdfff <unknown>

famin  v31.2s, v0.2s, v0.2s
// CHECK-INST: famin  v31.2s, v0.2s, v0.2s
// CHECK-ENCODING: [0x1f,0xdc,0xa0,0x2e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 2ea0dc1f <unknown>

famin  v0.2s, v0.2s, v0.2s
// CHECK-INST: famin  v0.2s, v0.2s, v0.2s
// CHECK-ENCODING: [0x00,0xdc,0xa0,0x2e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 2ea0dc00 <unknown>

famin  v31.4s, v31.4s, v31.4s
// CHECK-INST: famin  v31.4s, v31.4s, v31.4s
// CHECK-ENCODING: [0xff,0xdf,0xbf,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6ebfdfff <unknown>

famin  v0.4s, v31.4s, v31.4s
// CHECK-INST: famin  v0.4s, v31.4s, v31.4s
// CHECK-ENCODING: [0xe0,0xdf,0xbf,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6ebfdfe0 <unknown>

famin  v0.4s, v0.4s, v0.4s
// CHECK-INST: famin  v0.4s, v0.4s, v0.4s
// CHECK-ENCODING: [0x00,0xdc,0xa0,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6ea0dc00 <unknown>

famin  v31.2d, v31.2d, v31.2d
// CHECK-INST: famin  v31.2d, v31.2d, v31.2d
// CHECK-ENCODING: [0xff,0xdf,0xff,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6effdfff <unknown>

famin  v0.2d, v0.2d, v31.2d
// CHECK-INST: famin  v0.2d, v0.2d, v31.2d
// CHECK-ENCODING: [0x00,0xdc,0xff,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6effdc00 <unknown>

famin  v0.2d, v0.2d, v0.2d
// CHECK-INST: famin  v0.2d, v0.2d, v0.2d
// CHECK-ENCODING: [0x00,0xdc,0xe0,0x6e]
// CHECK-ERROR: instruction requires: faminmax
// CHECK-UNKNOWN: 6ee0dc00 <unknown>

