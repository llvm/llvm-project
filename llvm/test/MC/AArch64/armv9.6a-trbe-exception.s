// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



msr trbsr_el12, x0
// CHECK-INST: msr TRBSR_EL12, x0
// CHECK-ENCODING: encoding: [0x60,0x9b,0x1d,0xd5]
// CHECK-UNKNOWN:  d51d9b60 msr TRBSR_EL12, x0

msr trbsr_el2, x0
// CHECK-INST: msr TRBSR_EL2, x0
// CHECK-ENCODING: encoding: [0x60,0x9b,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c9b60 msr TRBSR_EL2, x0

msr trbsr_el3, x0
// CHECK-INST: msr TRBSR_EL3, x0
// CHECK-ENCODING: encoding: [0x60,0x9b,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e9b60 msr TRBSR_EL3, x0

mrs x0, trbsr_el12
// CHECK-INST: mrs x0, TRBSR_EL12
// CHECK-ENCODING: encoding: [0x60,0x9b,0x3d,0xd5]
// CHECK-UNKNOWN:  d53d9b60 mrs x0, TRBSR_EL12

mrs x0, trbsr_el2
// CHECK-INST: mrs x0, TRBSR_EL2
// CHECK-ENCODING: encoding: [0x60,0x9b,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c9b60 mrs x0, TRBSR_EL2

mrs x0, trbsr_el3
// CHECK-INST: mrs x0, TRBSR_EL3
// CHECK-ENCODING: encoding: [0x60,0x9b,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e9b60 mrs x0, TRBSR_EL3
