// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// Check that XAFlag/AXFlag don't accept operands like MSR does
xaflag S0_0_C4_C0_1, xzr
axflag S0_0_C4_C0_1, xzr

// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: xaflag S0_0_C4_C0_1, xzr
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: axflag S0_0_C4_C0_1, xzr
