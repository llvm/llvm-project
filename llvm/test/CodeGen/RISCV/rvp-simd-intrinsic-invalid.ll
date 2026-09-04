; RUN: sed -n '/^; BEGIN-PSEXT-LEGAL$/,/^; END-PSEXT-LEGAL$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+experimental-p,+m,+zbb \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PSEXT
; RUN: sed -n '/^; BEGIN-PSEXT-WIDEN$/,/^; END-PSEXT-WIDEN$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+experimental-p,+m,+zbb \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PSEXT
; RUN: sed -n '/^; BEGIN-PZEXT$/,/^; END-PZEXT$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+experimental-p,+m,+zbb \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PZEXT
; RUN: sed -n '/^; BEGIN-MUL-H$/,/^; END-MUL-H$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+m \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=MULPARTS
; RUN: sed -n '/^; BEGIN-MUL-W$/,/^; END-MUL-W$/p' %s \
; RUN:   | not llc -mtriple=riscv32 -mattr=+v,+m \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=MULPARTS

; PSEXT: LLVM ERROR: unsupported llvm.riscv.psext intrinsic
; PZEXT: LLVM ERROR: unsupported llvm.riscv.pzext intrinsic
; MULPARTS: LLVM ERROR: unsupported llvm.riscv multiply-parts intrinsic

; BEGIN-PSEXT-LEGAL
define <4 x i16> @bad_psext_legal_type(<4 x i16> %a) {
  %res = call <4 x i16> @llvm.riscv.psext.h.v4i16(<4 x i16> %a)
  ret <4 x i16> %res
}

declare <4 x i16> @llvm.riscv.psext.h.v4i16(<4 x i16>)
; END-PSEXT-LEGAL

; BEGIN-PSEXT-WIDEN
define <2 x i16> @bad_psext_widen_type(<2 x i16> %a) {
  %res = call <2 x i16> @llvm.riscv.psext.h.v2i16(<2 x i16> %a)
  ret <2 x i16> %res
}

declare <2 x i16> @llvm.riscv.psext.h.v2i16(<2 x i16>)
; END-PSEXT-WIDEN

; BEGIN-PZEXT
define <2 x i32> @bad_pzext(<2 x i32> %a) {
  %res = call <2 x i32> @llvm.riscv.pzext.b.v2i32(<2 x i32> %a)
  ret <2 x i32> %res
}

declare <2 x i32> @llvm.riscv.pzext.b.v2i32(<2 x i32>)
; END-PZEXT

; The multiply-parts intrinsics are only legalizable with the P extension.
; BEGIN-MUL-H
define i32 @bad_mul_h00_without_p(<2 x i16> %a, <2 x i16> %b) {
  %res = call i32 @llvm.riscv.mul.00.i32.v2i16(<2 x i16> %a, <2 x i16> %b)
  ret i32 %res
}

declare i32 @llvm.riscv.mul.00.i32.v2i16(<2 x i16>, <2 x i16>)
; END-MUL-H

; BEGIN-MUL-W
define i64 @bad_mul_w00_without_p(<2 x i32> %a, <2 x i32> %b) {
  %res = call i64 @llvm.riscv.mul.00.i64.v2i32(<2 x i32> %a, <2 x i32> %b)
  ret i64 %res
}

declare i64 @llvm.riscv.mul.00.i64.v2i32(<2 x i32>, <2 x i32>)
; END-MUL-W
