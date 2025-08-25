// +the required for RCWSMASK_EL1, RCWMASK_EL1
// +el2vmsa required for TTBR0_EL2 (VSCTLR_EL2), VTTBR_EL2
// +vh required for TTBR1_EL2

// RUN: not llvm-mc -triple=aarch64 -mattr=+d128,+the,+el2vmsa,+vh -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

msrr TTBR0_EL1, x0, x2
// CHECK-ERROR: error: expected second odd register of a consecutive same-size even/odd register pair

msrr TTBR0_EL1, x0
// CHECK-ERROR: error: expected comma

msrr TTBR0_EL1, x1, x2
// CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair

msrr TTBR0_EL1, x31, x0
// CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair

msrr TTBR0_EL1, xzr, x30
// CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair

msrr TTBR0_EL1, xzr
// CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair

msrr S3_0_c2_c0_1
// CHECK-ERROR: error: too few operands for instruction

msrr x0, x1, S3_0_c2_c0_1
// CHECK-ERROR: error: expected first even register of a consecutive same-size even/odd register pair
