; RUN: not llc -mtriple=mips64-unknown-linux -mcpu=mips64r6 -mattr=+micromips  %s 2>&1 | FileCheck %s --check-prefix=MICROMIPS64R6
; RUN: not llc -mtriple=mips64-unknown-linux -mcpu=mips64 -mattr=+micromips  %s 2>&1 | FileCheck %s --check-prefix=MICROMIPS64

; Test that microMIPS64(R6) is not supported.

; MICROMIPS64R6: LLVM ERROR: microMIPS64R6 is not supported
; MICROMIPS64: LLVM ERROR: microMIPS64 is not supported
