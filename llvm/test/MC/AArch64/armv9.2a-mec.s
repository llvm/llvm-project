// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mec < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mec < %s \
// RUN:        | llvm-objdump -d --mattr=+mec --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mec < %s \
// RUN:        | llvm-objdump -d --mattr=-mec --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mec < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+mec -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple aarch64 -disassemble < %s 2>&1 | FileCheck --check-prefix=CHECK-NO-MEC %s


mrs x0, MECIDR_EL2
// CHECK-INST: mrs x0, MECIDR_EL2
// CHECK-ENCODING: encoding: [0xe0,0xa8,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca8e0 mrs x0, S3_4_C10_C8_7

mrs x0, MECID_P0_EL2
// CHECK-INST: mrs x0, MECID_P0_EL2
// CHECK-ENCODING: encoding: [0x00,0xa8,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca800 mrs x0, S3_4_C10_C8_0

mrs x0, MECID_A0_EL2
// CHECK-INST: mrs x0, MECID_A0_EL2
// CHECK-ENCODING: encoding: [0x20,0xa8,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca820 mrs x0, S3_4_C10_C8_1

mrs x0, MECID_P1_EL2
// CHECK-INST: mrs x0, MECID_P1_EL2
// CHECK-ENCODING: encoding: [0x40,0xa8,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca840 mrs x0, S3_4_C10_C8_2

mrs x0, MECID_A1_EL2
// CHECK-INST: mrs x0, MECID_A1_EL2
// CHECK-ENCODING: encoding: [0x60,0xa8,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca860 mrs x0, S3_4_C10_C8_3

mrs x0, VMECID_P_EL2
// CHECK-INST: mrs x0, VMECID_P_EL2
// CHECK-ENCODING: encoding: [0x00,0xa9,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca900 mrs x0, S3_4_C10_C9_0

mrs x0, VMECID_A_EL2
// CHECK-INST: mrs x0, VMECID_A_EL2
// CHECK-ENCODING: encoding: [0x20,0xa9,0x3c,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53ca920 mrs x0, S3_4_C10_C9_1

mrs x0, MECID_RL_A_EL3
// CHECK-INST: mrs x0, MECID_RL_A_EL3
// CHECK-ENCODING: encoding: [0x20,0xaa,0x3e,0xd5]
// CHECK-ERROR: error: expected readable system register
// CHECK-UNKNOWN:  d53eaa20 mrs x0, S3_6_C10_C10_1

msr MECID_P0_EL2,    x0
// CHECK-INST: msr MECID_P0_EL2, x0
// CHECK-ENCODING: encoding: [0x00,0xa8,0x1c,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51ca800 msr S3_4_C10_C8_0, x0

msr MECID_A0_EL2,    x0
// CHECK-INST: msr MECID_A0_EL2, x0
// CHECK-ENCODING: encoding: [0x20,0xa8,0x1c,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51ca820 msr S3_4_C10_C8_1, x0

msr MECID_P1_EL2,    x0
// CHECK-INST: msr MECID_P1_EL2, x0
// CHECK-ENCODING: encoding: [0x40,0xa8,0x1c,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51ca840 msr S3_4_C10_C8_2, x0

msr MECID_A1_EL2,    x0
// CHECK-INST: msr MECID_A1_EL2, x0
// CHECK-ENCODING: encoding: [0x60,0xa8,0x1c,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51ca860 msr S3_4_C10_C8_3, x0

msr VMECID_P_EL2,   x0
// CHECK-INST: msr VMECID_P_EL2, x0
// CHECK-ENCODING: encoding: [0x00,0xa9,0x1c,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51ca900 msr S3_4_C10_C9_0, x0

msr VMECID_A_EL2,   x0
// CHECK-INST: msr VMECID_A_EL2, x0
// CHECK-ENCODING: encoding: [0x20,0xa9,0x1c,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51ca920 msr S3_4_C10_C9_1, x0

msr MECID_RL_A_EL3, x0
// CHECK-INST: msr MECID_RL_A_EL3, x0
// CHECK-ENCODING: encoding: [0x20,0xaa,0x1e,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51eaa20 msr S3_6_C10_C10_1, x0

dc cigdpae, x0
// CHECK-INST: dc cigdpae, x0
// CHECK-ENCODING: encoding: [0xe0,0x7e,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:4: error: DC CIGDPAE requires: mec
// CHECK-UNKNOWN:  d50c7ee0 sys #4, c7, c14, #7, x0
// CHECK-NO-MEC: sys #4, c7, c14, #7, x0

dc cipae, x0
// CHECK-INST: dc cipae, x0
// CHECK-ENCODING: encoding: [0x00,0x7e,0x0c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:4: error: DC CIPAE requires: mec
// CHECK-UNKNOWN:  d50c7e00 sys #4, c7, c14, #0, x0
// CHECK-NO-MEC: sys #4, c7, c14, #0, x0

sys #4, c7, c14, #7, x0
// CHECK-INST: dc cigdpae, x0
// CHECK-ENCODING: encoding: [0xe0,0x7e,0x0c,0xd5]
// CHECK-UNKNOWN:  d50c7ee0 sys #4, c7, c14, #7, x0

sys #4, c7, c14, #0, x0
// CHECK-INST: dc cipae, x0
// CHECK-ENCODING: encoding: [0x00,0x7e,0x0c,0xd5]
// CHECK-UNKNOWN:  d50c7e00 sys #4, c7, c14, #0, x0
