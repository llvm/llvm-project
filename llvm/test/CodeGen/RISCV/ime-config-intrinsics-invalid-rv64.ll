; REQUIRES: riscv-registered-target
; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=riscv64 -mattr=+experimental-zvvmm < %t/zero.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not --crash llc -mtriple=riscv64 -mattr=+experimental-zvvmm < %t/three.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not --crash llc -mtriple=riscv64 -mattr=+experimental-zvvmm < %t/too-large.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE

; BAD-VALUE: invalid constant requested lambda for llvm.riscv.ime.vsetlambda.nonzero

;--- zero.ll
define i64 @vsetlambda_invalid_zero() {
  %lambda = call i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64 0)
  ret i64 %lambda
}

declare i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64)

;--- three.ll
define i64 @vsetlambda_invalid_three() {
  %lambda = call i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64 3)
  ret i64 %lambda
}

declare i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64)

;--- too-large.ll
define i64 @vsetlambda_invalid_128() {
  %lambda = call i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64 128)
  ret i64 %lambda
}

declare i64 @llvm.riscv.ime.vsetlambda.nonzero.i64(i64)
