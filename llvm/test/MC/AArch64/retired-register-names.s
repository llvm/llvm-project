// RUN: not llvm-mc -triple=aarch64 -mattr=+all < %s 2>&1 | FileCheck %s

// Add negative tests preventing reuse of retired system register names.
//
// When system registers are retired, they should be appended to
// this file, to ensure they're never re-used in future.

mrs x0, mpamvidcr_el2
// CHECK:      :[[@LINE-1]]:9: error: expected readable system register
// CHECK-NEXT: mrs x0, mpamvidcr_el2

mrs x0, mpamvidsr_el2
// CHECK:      :[[@LINE-1]]:9: error: expected readable system register
// CHECK-NEXT: mrs x0, mpamvidsr_el2

mrs x0, mpamvidsr_el3
// CHECK:      :[[@LINE-1]]:9: error: expected readable system register
// CHECK-NEXT: mrs x0, mpamvidsr_el3
