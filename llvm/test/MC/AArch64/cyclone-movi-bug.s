; RUN: llvm-mc -triple aarch64-apple-ios -mcpu=cyclone %s 2> %t.log | FileCheck %s
; RUN: FileCheck %s --check-prefix=CHECK-ERR < %t.log

    ; CHECK: movi.16b v3, #0
    ; CHECK: movi.16b v7, #0
    ; CHECK-ERR: warning: instruction movi.2d with immediate #0 may not function correctly on this CPU, converting to equivalent movi.16b
    ; CHECK-ERR: warning: instruction movi.2d with immediate #0 may not function correctly on this CPU, converting to equivalent movi.16b
    movi.2d v3, #0
    movi v7.2d, #0
