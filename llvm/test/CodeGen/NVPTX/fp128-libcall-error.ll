; RUN: split-file %s %t
; RUN: not llc < %t/mul.ll -mtriple=nvptx64 -filetype=null 2>&1 | FileCheck %s --check-prefix=FMUL
; RUN: not llc < %t/sqrt.ll -mtriple=nvptx64 -filetype=null 2>&1 | FileCheck %s --check-prefix=FSQRT

; FMUL: error: no libcall available for fmul
; FSQRT: error: no libcall available for fsqrt

;--- mul.ll
define fp128 @mul128(fp128 %a) {
  %r = fmul fp128 %a, 0xL00000000000000004000400000000000
  ret fp128 %r
}

;--- sqrt.ll
define fp128 @sqrt128(fp128 %a) {
  %r = call fp128 @llvm.sqrt.f128(fp128 %a)
  ret fp128 %r
}

declare fp128 @llvm.sqrt.f128(fp128)
