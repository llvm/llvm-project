// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+specrestrict < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a        < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-specrestrict < %s 2>&1 | FileCheck %s --check-prefix=NOSPECID

mrs x9, ID_PFR2_EL1

// CHECK:         mrs x9, {{id_pfr2_el1|ID_PFR2_EL1}}            // encoding: [0x89,0x03,0x38,0xd5]
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x9, ID_PFR2_EL1

mrs x8, SCXTNUM_EL0
mrs x7, SCXTNUM_EL1
mrs x6, SCXTNUM_EL2
mrs x5, SCXTNUM_EL3
mrs x4, SCXTNUM_EL12

// CHECK:         mrs x8, {{scxtnum_el0|SCXTNUM_EL0}}            // encoding: [0xe8,0xd0,0x3b,0xd5]
// CHECK:         mrs x7, {{scxtnum_el1|SCXTNUM_EL1}}            // encoding: [0xe7,0xd0,0x38,0xd5]
// CHECK:         mrs x6, {{scxtnum_el2|SCXTNUM_EL2}}            // encoding: [0xe6,0xd0,0x3c,0xd5]
// CHECK:         mrs x5, {{scxtnum_el3|SCXTNUM_EL3}}            // encoding: [0xe5,0xd0,0x3e,0xd5]
// CHECK:         mrs x4, {{scxtnum_el12|SCXTNUM_EL12}}          // encoding: [0xe4,0xd0,0x3d,0xd5]
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x8, {{scxtnum_el0|SCXTNUM_EL0}}
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x7, {{scxtnum_el1|SCXTNUM_EL1}}
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x6, {{scxtnum_el2|SCXTNUM_EL2}}
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x5, {{scxtnum_el3|SCXTNUM_EL3}}
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x4, {{scxtnum_el12|SCXTNUM_EL12}}

msr SCXTNUM_EL0,  x8
msr SCXTNUM_EL1,  x7
msr SCXTNUM_EL2,  x6
msr SCXTNUM_EL3,  x5
msr SCXTNUM_EL12, x4

// CHECK:         msr {{scxtnum_el0|SCXTNUM_EL0}},   x8          // encoding: [0xe8,0xd0,0x1b,0xd5]
// CHECK:         msr {{scxtnum_el1|SCXTNUM_EL1}},   x7          // encoding: [0xe7,0xd0,0x18,0xd5]
// CHECK:         msr {{scxtnum_el2|SCXTNUM_EL2}},   x6          // encoding: [0xe6,0xd0,0x1c,0xd5]
// CHECK:         msr {{scxtnum_el3|SCXTNUM_EL3}},   x5          // encoding: [0xe5,0xd0,0x1e,0xd5]
// CHECK:         msr {{scxtnum_el12|SCXTNUM_EL12}}, x4          // encoding: [0xe4,0xd0,0x1d,0xd5]
// NOSPECID:      error: expected writable system register
// NOSPECID-NEXT: {{scxtnum_el0|SCXTNUM_EL0}}
// NOSPECID:      error: expected writable system register
// NOSPECID-NEXT: {{scxtnum_el1|SCXTNUM_EL1}}
// NOSPECID:      error: expected writable system register
// NOSPECID-NEXT: {{scxtnum_el2|SCXTNUM_EL2}}
// NOSPECID:      error: expected writable system register
// NOSPECID-NEXT: {{scxtnum_el3|SCXTNUM_EL3}}
// NOSPECID:      error: expected writable system register
// NOSPECID-NEXT: {{scxtnum_el12|SCXTNUM_EL12}}
