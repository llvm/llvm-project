# RUN: not llvm-mc -triple i386-unknown-unknown %s 2>&1 > /dev/null| FileCheck %s --check-prefix=CHECK-ERROR

#CHECK-ERROR: error: instruction must have size higher than 0
    .byte 64;""
