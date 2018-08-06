// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// ARMV8.4-A Debug, Trace and PMU Extensions
//------------------------------------------------------------------------------

tsb
tsb foo
tsb #0
tsb 0

//CHECK-ERROR: error: too few operands for instruction
//CHECK-ERROR: tsb
//CHECK-ERROR: ^
//CHECK-ERROR: error: 'csync' operand expected
//CHECK-ERROR: tsb foo
//CHECK-ERROR:     ^
//CHECK-ERROR: error: 'csync' operand expected
//CHECK-ERROR: tsb #0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: 'csync' operand expected
//CHECK-ERROR: tsb 0
//CHECK-ERROR:     ^
