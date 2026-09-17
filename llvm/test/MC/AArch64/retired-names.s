// RUN: not llvm-mc -triple=aarch64 -mattr=+mpamv2,+tme < %s 2>&1 | FileCheck %s

// Retired feature and system register names must not be reused.

// CHECK:      '+mpamv2' is not a recognized feature for this target (ignoring feature)
// CHECK-NEXT: '+tme' is not a recognized feature for this target (ignoring feature)

mrs x0, mpamvidcr_el2
// CHECK:      :[[@LINE-1]]:9: error: expected readable system register
// CHECK-NEXT: mrs x0, mpamvidcr_el2

mrs x0, mpamvidsr_el2
// CHECK:      :[[@LINE-1]]:9: error: expected readable system register
// CHECK-NEXT: mrs x0, mpamvidsr_el2

mrs x0, mpamvidsr_el3
// CHECK:      :[[@LINE-1]]:9: error: expected readable system register
// CHECK-NEXT: mrs x0, mpamvidsr_el3
