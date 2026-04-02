// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



//------------------------------------------------------------------------------
// Armv9.6-A FEAT_MPAM Extensions
//------------------------------------------------------------------------------

msr MPAMBW3_EL3, x0
// CHECK-INST: msr MPAMBW3_EL3, x0
// CHECK-ENCODING: encoding: [0x80,0xa5,0x1e,0xd5]
// CHECK-UNKNOWN:  d51ea580 msr MPAMBW3_EL3, x0

msr MPAMBW2_EL2, x0
// CHECK-INST: msr MPAMBW2_EL2, x0
// CHECK-ENCODING: encoding: [0x80,0xa5,0x1c,0xd5]
// CHECK-UNKNOWN:  d51ca580 msr MPAMBW2_EL2, x0

msr MPAMBW1_EL1, x0
// CHECK-INST: msr MPAMBW1_EL1, x0
// CHECK-ENCODING: encoding: [0x80,0xa5,0x18,0xd5]
// CHECK-UNKNOWN:  d518a580 msr MPAMBW1_EL1, x0

msr MPAMBW1_EL12, x0
// CHECK-INST: msr MPAMBW1_EL12, x0
// CHECK-ENCODING: encoding: [0x80,0xa5,0x1d,0xd5]
// CHECK-UNKNOWN:  d51da580 msr MPAMBW1_EL12, x0

msr MPAMBW0_EL1, x0
// CHECK-INST: msr MPAMBW0_EL1, x0
// CHECK-ENCODING: encoding: [0xa0,0xa5,0x18,0xd5]
// CHECK-UNKNOWN:  d518a5a0 msr MPAMBW0_EL1, x0

msr MPAMBWCAP_EL2, x0
// CHECK-INST: msr MPAMBWCAP_EL2, x0
// CHECK-ENCODING: encoding: [0xc0,0xa5,0x1c,0xd5]
// CHECK-UNKNOWN:  d51ca5c0 msr MPAMBWCAP_EL2, x0

msr MPAMBWSM_EL1, x0
// CHECK-INST: msr MPAMBWSM_EL1, x0
// CHECK-ENCODING: encoding: [0xe0,0xa5,0x18,0xd5]
// CHECK-UNKNOWN:  d518a5e0 msr MPAMBWSM_EL1, x0

mrs x0, MPAMBWIDR_EL1
// CHECK-INST: mrs x0, MPAMBWIDR_EL1
// CHECK-ENCODING: encoding: [0xa0,0xa4,0x38,0xd5]
// CHECK-UNKNOWN:  d538a4a0 mrs x0, MPAMBWIDR_EL1

mrs x0, MPAMBW3_EL3
// CHECK-INST: mrs x0, MPAMBW3_EL3
// CHECK-ENCODING: encoding: [0x80,0xa5,0x3e,0xd5]
// CHECK-UNKNOWN:  d53ea580 mrs x0, MPAMBW3_EL3

mrs x0, MPAMBW2_EL2
// CHECK-INST: mrs x0, MPAMBW2_EL2
// CHECK-ENCODING: encoding: [0x80,0xa5,0x3c,0xd5]
// CHECK-UNKNOWN:  d53ca580 mrs x0, MPAMBW2_EL2

mrs x0, MPAMBW1_EL1
// CHECK-INST: mrs x0, MPAMBW1_EL1
// CHECK-ENCODING: encoding: [0x80,0xa5,0x38,0xd5]
// CHECK-UNKNOWN:  d538a580 mrs x0, MPAMBW1_EL1

mrs x0, MPAMBW1_EL12
// CHECK-INST: mrs x0, MPAMBW1_EL12
// CHECK-ENCODING: encoding: [0x80,0xa5,0x3d,0xd5]
// CHECK-UNKNOWN:  d53da580 mrs x0, MPAMBW1_EL12

mrs x0, MPAMBW0_EL1
// CHECK-INST: mrs x0, MPAMBW0_EL1
// CHECK-ENCODING: encoding: [0xa0,0xa5,0x38,0xd5]
// CHECK-UNKNOWN:  d538a5a0 mrs x0, MPAMBW0_EL1

mrs x0, MPAMBWCAP_EL2
// CHECK-INST: mrs x0, MPAMBWCAP_EL2
// CHECK-ENCODING: encoding: [0xc0,0xa5,0x3c,0xd5]
// CHECK-UNKNOWN:  d53ca5c0 mrs x0, MPAMBWCAP_EL2

mrs x0, MPAMBWSM_EL1
// CHECK-INST: mrs x0, MPAMBWSM_EL1
// CHECK-ENCODING: encoding: [0xe0,0xa5,0x38,0xd5]
// CHECK-UNKNOWN:  d538a5e0 mrs x0, MPAMBWSM_EL1


