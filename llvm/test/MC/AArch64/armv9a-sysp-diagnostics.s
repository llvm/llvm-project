// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

tlbip ALLE1
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE1IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE1ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE1NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE1OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE1OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE2
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE2IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE2ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE2NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE2OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE2OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE3
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE3IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE3ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE3NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE3OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ALLE3OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ASIDE1
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ASIDE1IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ASIDE1ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ASIDE1NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ASIDE1OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip ASIDE1OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip PAALL
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip PAALLOS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip RPALOS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip RPAOS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLE1
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLE1IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLE1ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLE1NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLE1OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLE1OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLS12E1
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLS12E1IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLS12E1ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLS12E1NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLS12E1OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLS12E1OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLWS2E1
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLWS2E1IS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLWS2E1ISNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLWS2E1NXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLWS2E1OS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
tlbip VMALLWS2E1OSNXS
// CHECK-ERROR: error: invalid operand for TLBIP instruction
