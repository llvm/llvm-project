// RUN: not llvm-mc -triple arm64-- -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-GENERIC < %t %s

// RUN: llvm-mc -triple arm64e-- -show-encoding < %s |\
// RUN: FileCheck %s --check-prefix=CHECK-ARM64E

// CHECK-GENERIC:  error: instruction requires: pa
// CHECK-ARM64E:  pacia x0, x1 // encoding: [0x20,0x00,0xc1,0xda]
  pacia x0, x1
