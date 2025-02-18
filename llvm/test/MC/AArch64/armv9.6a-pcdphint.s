// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+pcdphint %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding %s 2>&1 | FileCheck --check-prefix=ERROR %s

.func:
// CHECK: .func:
  stshh keep
// CHECK: stshh	keep                            // encoding: [0x1f,0x96,0x01,0xd5]
// ERROR: error: instruction requires: pcdphint
  stshh strm
// CHECK: stshh	strm                            // encoding: [0x3f,0x96,0x01,0xd5]
// ERROR: error: instruction requires: pcdphint


