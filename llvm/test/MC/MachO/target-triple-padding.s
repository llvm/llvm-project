// REQUIRES: aarch64-registered-target

// RUN: split-file %s %t

// RUN: llvm-mc -triple arm64-ios-macabi %t/target-triple-always-has-nul.s -filetype=obj -o - \
// RUN: | llvm-objdump --macho --private-headers - | FileCheck %t/target-triple-always-has-nul.s

// RUN: llvm-mc -triple arm64-apple-driverkit %t/target-triple-no-extra-padding.s -filetype=obj -o - \
// RUN: | llvm-objdump --macho --private-headers - | FileCheck %t/target-triple-no-extra-padding.s

//--- target-triple-always-has-nul.s
// Make sure a target triple that reaches an 8 byte boundary doesn't drop its NUL.
// LC_TARGET_TRIPLE is 3 32-bit integers (12 bytes) plus the target triple string.
// A target triple length of 28 hits the 8 byte boundary (12 + 28 = 40). However the
// NUL byte on the end of the string is required, so it should pad out to 48 bytes.
.target_triple "arm64-apple-ios27.0.0-macabi"

// CHECK:           cmd LC_TARGET_TRIPLE
// CHECK-NEXT:  cmdsize 48
// CHECK-NEXT:   triple arm64-apple-ios27.0.0-macabi

//--- target-triple-no-extra-padding.s
// A target triple length of 27 perfectly fits in an 8 byte boundary when including
// the string's NUL byte. That should come out to 40 bytes with no extra padding.
.target_triple "arm64-apple-driverkit27.0.0"

// CHECK:           cmd LC_TARGET_TRIPLE
// CHECK-NEXT:  cmdsize 40
// CHECK-NEXT:   triple arm64-apple-driverkit27.0.0
