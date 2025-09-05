// CLRBHB is optional for all v8a/v9a, mandatory for 8.9a/9.4a.
// Assembly is always permitted for instructions in the hint space.

// Optional, off by default
// RUN: llvm-mc -show-encoding -triple aarch64 < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8a < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8.8a < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9a < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9.3a < %s | FileCheck %s --check-prefix=HINT_22

// Optional, off by default, doubly disabled
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8.8a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9.3a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22

// Optional, off by default, manually enabled
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8a,+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8.8a,+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9a,+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9.3a,+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Mandatory, enabled by default
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8.9a < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9.4a < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Mandatory, on by default, doubly enabled
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8.9a,+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9.4a,+clrbhb < %s | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Mandatory, can't prevent disabling in LLVM
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v8.9a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22
// RUN: llvm-mc -show-encoding -triple aarch64 -mattr=+v9.4a,-clrbhb < %s | FileCheck %s --check-prefix=HINT_22

// Check Unknown
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+clrbhb < %s \
// RUN:   | llvm-objdump -d --mattr=-clrbhb --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+clrbhb < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+clrbhb -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

clrbhb
// HINT_22: hint #22                             // encoding: [0xdf,0x22,0x03,0xd5]
// CHECK-INST: clrbhb
// CHECK-ENCODING: encoding: [0xdf,0x22,0x03,0xd5]
// CHECK-UNKNOWN:  d50322df    hint #22

hint #22
// HINT_22: hint #22                             // encoding: [0xdf,0x22,0x03,0xd5]
// CHECK-INST: clrbhb
// CHECK-ENCODING: encoding: [0xdf,0x22,0x03,0xd5]
// CHECK-UNKNOWN:  d50322df    hint #22
