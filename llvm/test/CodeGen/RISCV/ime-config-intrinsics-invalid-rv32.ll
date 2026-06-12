; REQUIRES: riscv-registered-target
; RUN: not --crash llc -mtriple=riscv32 -mattr=+experimental-zvvmm < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE

; BAD-VALUE: invalid argument for llvm.riscv.ime.vsetlambda.nonzero

define i32 @vsetlambda_invalid_zero() {
  %lambda = call i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32 0)
  ret i32 %lambda
}

declare i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32 immarg)
