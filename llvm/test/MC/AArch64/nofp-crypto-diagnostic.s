// RUN: not llvm-mc  -triple aarch64-none-linux-gnu -mattr=+neon,+crypto,-fp-armv8 < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

        sha1h s0, s1

// CHECK-ERROR: error: instruction requires: sha2
// CHECK-ERROR-NEXT:    sha1h s0, s1
// CHECK-ERROR-NEXT:    ^
