// RUN: not llvm-mc -triple arm-apple-darwin %s 2> %t
// RUN: FileCheck -input-file %t %s

// CHECK: error: brackets expression not supported on this target
.byte	[4-3]
