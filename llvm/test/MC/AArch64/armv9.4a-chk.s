// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v8.9a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v9.4a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+chk < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+chk < %s \
// RUN:        | llvm-objdump -d --mattr=+chk - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+chk < %s \
// RUN:   | llvm-objdump -d --mattr=-chk - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+chk < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+chk -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// FEAT_CHK is mandatory from v8.0-a, but a clang user may not be using the LLVM
// integrated assembler, so we cannot just print `chkfeat x16` in all
// circumstances. Thankfully, we can always print `hint #40` when we cannot
// print `chkfeat x16`.
// So, in this case, we only print `chkfeat x16` from v8.9-a onwards, as an
// assembler that understands v8.9-a will understand `chkfeat x16`, and those
// that understand previous versions may not.

chkfeat x16
// CHECK-INST: chkfeat x16
// CHECK-ENCODING: encoding: [0x1f,0x25,0x03,0xd5]
// CHECK-ERROR: hint #40
// CHECK-UNKNOWN:  d503251f      hint #40

hint #40
// CHECK-INST: chkfeat x16
// CHECK-ENCODING: encoding: [0x1f,0x25,0x03,0xd5]
// CHECK-ERROR: hint #40
// CHECK-UNKNOWN:  d503251f      hint #40
