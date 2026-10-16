; RUN: split-file %s %t
; RUN: not llc < %t/float8e4m3.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E4M3
; RUN: not llc < %t/float8e3m4.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E3M4
; RUN: not llc < %t/float8e5m2fnuz.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E5M2FNUZ
; RUN: not llc < %t/float8e4m3fnuz.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E4M3FNUZ
; RUN: not llc < %t/float8e4m3b11fnuz.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E4M3B11FNUZ
; RUN: not llc < %t/float8e8m0fnu.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E8M0FNU

; Test that llvm.convert.from.arbitrary.fp emits an error for formats that pass
; verifier validation but are not yet implemented in SelectionDAGBuilder.

;--- float8e4m3.ll
; E4M3: error: convert_from_arbitrary_fp: not implemented format 'Float8E4M3'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e4m3(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E4M3")
  ret float %r
}

;--- float8e3m4.ll
; E3M4: error: convert_from_arbitrary_fp: not implemented format 'Float8E3M4'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e3m4(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E3M4")
  ret float %r
}

;--- float8e5m2fnuz.ll
; E5M2FNUZ: error: convert_from_arbitrary_fp: not implemented format 'Float8E5M2FNUZ'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e5m2fnuz(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E5M2FNUZ")
  ret float %r
}

;--- float8e4m3fnuz.ll
; E4M3FNUZ: error: convert_from_arbitrary_fp: not implemented format 'Float8E4M3FNUZ'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e4m3fnuz(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E4M3FNUZ")
  ret float %r
}

;--- float8e4m3b11fnuz.ll
; E4M3B11FNUZ: error: convert_from_arbitrary_fp: not implemented format 'Float8E4M3B11FNUZ'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e4m3b11fnuz(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E4M3B11FNUZ")
  ret float %r
}

;--- float8e8m0fnu.ll
; E8M0FNU: error: convert_from_arbitrary_fp: not implemented format 'Float8E8M0FNU'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e8m0fnu(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E8M0FNU")
  ret float %r
}
