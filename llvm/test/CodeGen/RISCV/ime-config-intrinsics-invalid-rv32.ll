; REQUIRES: riscv-registered-target
; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+experimental-zvvmm < %t/zero.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not llc -mtriple=riscv32 -mattr=+experimental-zvvmm < %t/three.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not llc -mtriple=riscv32 -mattr=+experimental-zvvmm < %t/too-large.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-VALUE

; BAD-VALUE: invalid constant requested lambda for llvm.riscv.ime.vsetlambda.nonzero

;--- zero.ll
define i32 @vsetlambda_invalid_zero() {
  %lambda = call i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32 0)
  ret i32 %lambda
}

declare i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32)

;--- three.ll
define i32 @vsetlambda_invalid_three() {
  %lambda = call i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32 3)
  ret i32 %lambda
}

declare i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32)

;--- too-large.ll
define i32 @vsetlambda_invalid_128() {
  %lambda = call i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32 128)
  ret i32 %lambda
}

declare i32 @llvm.riscv.ime.vsetlambda.nonzero.i32(i32)
