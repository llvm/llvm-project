@ RUN: not llvm-mc -triple=armv7-apple-darwin -mattr=-neon < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

        vmov d5, d10
        vmov q4, q5
@ CHECK-ERRORS: error: instruction requires: NEON
@ CHECK-ERRORS: error: instruction requires: NEON
