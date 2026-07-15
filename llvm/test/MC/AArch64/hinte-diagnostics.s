// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+hinte < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR

hinte #-1
// CHECK-ERROR: error: immediate must be an integer in range [0, 65535], excluding values in range [12319, 16383] where (value - 12319) is a multiple of 32.

hinte #65536
// CHECK-ERROR: error: immediate must be an integer in range [0, 65535], excluding values in range [12319, 16383] where (value - 12319) is a multiple of 32.

hinte #12319
// CHECK-ERROR: error: immediate must be an integer in range [0, 65535], excluding values in range [12319, 16383] where (value - 12319) is a multiple of 32.

hinte #12383
// CHECK-ERROR: error: immediate must be an integer in range [0, 65535], excluding values in range [12319, 16383] where (value - 12319) is a multiple of 32.

hinte #16383
// CHECK-ERROR: error: immediate must be an integer in range [0, 65535], excluding values in range [12319, 16383] where (value - 12319) is a multiple of 32.
