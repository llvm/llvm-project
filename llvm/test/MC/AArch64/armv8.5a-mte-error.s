// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+mte < %s 2>&1| FileCheck %s

mrs tco
mrs gcr_el1
mrs rgsr_el1
mrs tfsr_el1
mrs tfsr_el2
mrs tfsr_el3
mrs tfsr_el12
mrs tfsre0_el1

// CHECK:      invalid operand for instruction
// CHECK-NEXT: tco
// CHECK:      invalid operand for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsre0_el1

mrs tco, #0
mrs tco, x0
mrs gcr_el1, x1
mrs rgsr_el1, x2
mrs tfsr_el1, x3
mrs tfsr_el2, x4
mrs tfsr_el3, x5
mrs tfsr_el12, x6
mrs tfsre0_el1, x7

// CHECK:      invalid operand for instruction
// CHECK-NEXT: tco, #0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tco, x0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsre0_el1

msr tco
msr gcr_el1
msr rgsr_el1
msr tfsr_el1
msr tfsr_el2
msr tfsr_el3
msr tfsr_el12
msr tfsre0_el1

// CHECK:      too few operands for instruction
// CHECK-NEXT: tco
// CHECK:      too few operands for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      too few operands for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsre0_el1

msr x0, tco
msr x1, gcr_el1
msr x2, rgsr_el1
msr x3, tfsr_el1
msr x4, tfsr_el2
msr x5, tfsr_el3
msr x6, tfsr_el12
msr x7, tfsre0_el1

// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tco
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: gcr_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: rgsr_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el2
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el3
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el12
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsre0_el1

// Among the system registers added by MTE, only TCO can be used with MSR (imm).
// The rest can only be used with MSR (reg).
msr gcr_el1, #1
msr rgsr_el1, #2
msr tfsr_el1, #3
msr tfsr_el2, #4
msr tfsr_el3, #5
msr tfsr_el12, #6
msr tfsre0_el1, #7

// CHECK:      invalid operand for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsre0_el1
