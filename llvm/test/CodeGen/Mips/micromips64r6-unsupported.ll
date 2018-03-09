; RUN: not llc -mtriple=mips64-unknown-linux -mcpu=mips64r6 -mattr=+micromips  %s 2>&1 | FileCheck %s

; Test that microMIPS64R6 is not supported.

; CHECK: LLVM ERROR: microMIPS64R6 is not supported
