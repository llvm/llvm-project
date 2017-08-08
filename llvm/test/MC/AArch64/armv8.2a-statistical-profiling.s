// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+spe < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s 2>&1 | FileCheck --check-prefix=NO_SPE %s

  psb csync
// CHECK: psb csync              // encoding: [0x3f,0x22,0x03,0xd5]
// NO_SPE:  invalid operand for instruction

  msr pmblimitr_el1, x0
  msr pmbptr_el1, x0
  msr pmbsr_el1, x0
  msr pmbidr_el1, x0
  msr pmscr_el2, x0
  msr pmscr_el12, x0
  msr pmscr_el1, x0
  msr pmsicr_el1, x0
  msr pmsirr_el1, x0
  msr pmsfcr_el1, x0
  msr pmsevfr_el1, x0
  msr pmslatfr_el1, x0
  msr pmsidr_el1, x0
// CHECK:     msr PMBLIMITR_EL1, x0       // encoding: [0x00,0x9a,0x18,0xd5]
// CHECK:     msr PMBPTR_EL1, x0          // encoding: [0x20,0x9a,0x18,0xd5]
// CHECK:     msr PMBSR_EL1, x0           // encoding: [0x60,0x9a,0x18,0xd5]
// CHECK:     msr PMBIDR_EL1, x0          // encoding: [0xe0,0x9a,0x18,0xd5]
// CHECK:     msr PMSCR_EL2, x0           // encoding: [0x00,0x99,0x1c,0xd5]
// CHECK:     msr PMSCR_EL12, x0          // encoding: [0x00,0x99,0x1d,0xd5]
// CHECK:     msr PMSCR_EL1, x0           // encoding: [0x00,0x99,0x18,0xd5]
// CHECK:     msr PMSICR_EL1, x0          // encoding: [0x40,0x99,0x18,0xd5]
// CHECK:     msr PMSIRR_EL1, x0          // encoding: [0x60,0x99,0x18,0xd5]
// CHECK:     msr PMSFCR_EL1, x0          // encoding: [0x80,0x99,0x18,0xd5]
// CHECK:     msr PMSEVFR_EL1, x0         // encoding: [0xa0,0x99,0x18,0xd5]
// CHECK:     msr PMSLATFR_EL1, x0        // encoding: [0xc0,0x99,0x18,0xd5]
// CHECK:     msr PMSIDR_EL1, x0          // encoding: [0xe0,0x99,0x18,0xd5]
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate
// NO_SPE: error: expected writable system register or pstate

mrs x0, pmblimitr_el1
  mrs x0, pmbptr_el1
  mrs x0, pmbsr_el1
  mrs x0, pmbidr_el1
  mrs x0, pmscr_el2
  mrs x0, pmscr_el12
  mrs x0, pmscr_el1
  mrs x0, pmsicr_el1
  mrs x0, pmsirr_el1
  mrs x0, pmsfcr_el1
  mrs x0, pmsevfr_el1
  mrs x0, pmslatfr_el1
  mrs x0, pmsidr_el1

// CHECK:    mrs x0, PMBLIMITR_EL1       // encoding: [0x00,0x9a,0x38,0xd5]
// CHECK:    mrs x0, PMBPTR_EL1          // encoding: [0x20,0x9a,0x38,0xd5]
// CHECK:    mrs x0, PMBSR_EL1           // encoding: [0x60,0x9a,0x38,0xd5]
// CHECK:    mrs x0, PMBIDR_EL1          // encoding: [0xe0,0x9a,0x38,0xd5]
// CHECK:    mrs x0, PMSCR_EL2           // encoding: [0x00,0x99,0x3c,0xd5]
// CHECK:    mrs x0, PMSCR_EL12          // encoding: [0x00,0x99,0x3d,0xd5]
// CHECK:    mrs x0, PMSCR_EL1           // encoding: [0x00,0x99,0x38,0xd5]
// CHECK:    mrs x0, PMSICR_EL1          // encoding: [0x40,0x99,0x38,0xd5]
// CHECK:    mrs x0, PMSIRR_EL1          // encoding: [0x60,0x99,0x38,0xd5]
// CHECK:    mrs x0, PMSFCR_EL1          // encoding: [0x80,0x99,0x38,0xd5]
// CHECK:    mrs x0, PMSEVFR_EL1         // encoding: [0xa0,0x99,0x38,0xd5]
// CHECK:    mrs x0, PMSLATFR_EL1        // encoding: [0xc0,0x99,0x38,0xd5]
// CHECK:    mrs x0, PMSIDR_EL1          // encoding: [0xe0,0x99,0x38,0xd5]
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
// NO_SPE: error: expected readable system register
