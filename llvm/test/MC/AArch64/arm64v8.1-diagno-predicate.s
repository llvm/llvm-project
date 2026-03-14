// RUN: not llvm-mc  -triple=arm64-linux-gnu -mattr=armv8.1a -mattr=-lse < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

        casa  w5, w7, [x20]
// CHECK-ERROR: error: instruction requires: lse
// CHECK-ERROR-NEXT:        casa  w5, w7, [x20]
// CHECK-ERROR-NEXT:        ^

