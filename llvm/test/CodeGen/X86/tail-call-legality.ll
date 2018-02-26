; RUN: llc -mtriple=i686-- -o - < %s | FileCheck %s

; This used to be classified as a tail call because of a mismatch in the
; arguments seen by Analysis.cpp and ISelLowering. As seen by ISelLowering, they
; both return {i32, i32, i32} (since i64 is illegal) which is fine for a tail
; call.

; As seen by Analysis.cpp: i64 -> i32 is a valid trunc, second i32 passes
; straight through and the third is undef, also OK for a tail call.

; Analysis.cpp was wrong.

; FIXME: in principle we *could* support some tail calls involving truncations
; of illegal types: a single "trunc i64 %whatever to i32" is probably valid
; because of how the extra registers are laid out.

declare {i64, i32} @test()

define {i32, i32, i32} @test_pair_notail(i64 %in) {
; CHECK-LABEL: test_pair_notail
; CHECK-NOT: jmp

  %whole = tail call {i64, i32} @test()
  %first = extractvalue {i64, i32} %whole, 0
  %first.trunc = trunc i64 %first to i32

  %second = extractvalue {i64, i32} %whole, 1

  %tmp = insertvalue {i32, i32, i32} undef, i32 %first.trunc, 0
  %res = insertvalue {i32, i32, i32} %tmp, i32 %second, 1
  ret {i32, i32, i32} %res
}
