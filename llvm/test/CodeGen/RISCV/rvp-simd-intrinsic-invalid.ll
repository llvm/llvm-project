; RUN: sed -n '/^; BEGIN-PSEXT-LEGAL$/,/^; END-PSEXT-LEGAL$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+experimental-p,+m,+zbb \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PSEXT
; RUN: sed -n '/^; BEGIN-PSEXT-WIDEN$/,/^; END-PSEXT-WIDEN$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+experimental-p,+m,+zbb \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PSEXT
; RUN: sed -n '/^; BEGIN-PZEXT$/,/^; END-PZEXT$/p' %s \
; RUN:   | not llc -mtriple=riscv64 -mattr=+experimental-p,+m,+zbb \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PZEXT

; PSEXT: LLVM ERROR: unsupported llvm.riscv.psext intrinsic
; PZEXT: LLVM ERROR: unsupported llvm.riscv.pzext intrinsic

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
