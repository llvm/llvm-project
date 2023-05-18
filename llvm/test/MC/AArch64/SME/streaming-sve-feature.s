// RUN: llvm-mc -triple=aarch64 -mattr=+sme < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -mattr=-neon,+sme < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// Verify NEON is disabled when targeting streaming mode, if it's not
// explicitly requested.
add v0.8b, v1.8b, v2.8b
// CHECK: add v0.8b, v1.8b, v2.8b
// CHECK-ERROR: error: instruction requires: neon
