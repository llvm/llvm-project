// RUN: llvm-mc -triple=aarch64 -mattr=+mpamv2,+tme < %s 2>&1 | FileCheck %s

// Add negative tests preventing reuse of retired feature names.
//
// When feature names are retired, they should be appended to
// this file, to ensure they're never re-used in future.

// CHECK:      '+mpamv2' is not a recognized feature for this target (ignoring feature)
// CHECK-NEXT: '+tme' is not a recognized feature for this target (ignoring feature)
