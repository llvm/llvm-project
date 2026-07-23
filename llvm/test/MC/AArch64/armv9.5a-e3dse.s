// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



mrs x0, VDISR_EL3
// CHECK-INST: mrs x0, VDISR_EL3
// CHECK-ENCODING: encoding: [0x20,0xc1,0x3e,0xd5]
// CHECK-UNKNOWN:  d53ec120 mrs x0, VDISR_EL3

msr VDISR_EL3, x0
// CHECK-INST: msr VDISR_EL3, x0
// CHECK-ENCODING: encoding: [0x20,0xc1,0x1e,0xd5]
// CHECK-UNKNOWN:  d51ec120 msr VDISR_EL3, x0

mrs x0, VSESR_EL3
// CHECK-INST: mrs x0, VSESR_EL3
// CHECK-ENCODING: encoding: [0x60,0x52,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e5260 mrs x0, VSESR_EL3

msr VSESR_EL3, x0
// CHECK-INST: msr VSESR_EL3, x0
// CHECK-ENCODING: encoding: [0x60,0x52,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e5260 msr VSESR_EL3, x0
