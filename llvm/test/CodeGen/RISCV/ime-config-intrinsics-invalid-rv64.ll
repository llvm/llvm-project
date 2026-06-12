; REQUIRES: riscv-registered-target
; RUN: not --crash llc -mtriple=riscv64 -mattr=+experimental-zvvmm < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE

; BAD-VALUE: invalid argument for llvm.riscv.ime.vsetlambda.nonzero

define i64 @vsetlambda_invalid_128() {
  %lambda = call i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64 128)
  ret i64 %lambda
}

declare i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64 immarg)
