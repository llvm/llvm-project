; RUN: opt -passes=vector-combine -verify-each -S %s | FileCheck %s

; The two operand groups contain already visited values, but neither group
; matches the field order of the deinterleave. They cannot be widened.
define <8 x i16> @duplicate_member(<8 x i64> %w) {
; CHECK-LABEL: @duplicate_member(
; CHECK: %r = call <8 x i16> @llvm.vector.interleave2.v8i16(<4 x i16> %x, <4 x i16> %y)
; CHECK: ret <8 x i16> %r
  %d = call { <4 x i64>, <4 x i64> } @llvm.vector.deinterleave2.v8i64(<8 x i64> %w)
  %a = extractvalue { <4 x i64>, <4 x i64> } %d, 0
  %b = extractvalue { <4 x i64>, <4 x i64> } %d, 1
  %ta = trunc <4 x i64> %a to <4 x i16>
  %tb = trunc <4 x i64> %b to <4 x i16>
  %x = sub <4 x i16> %ta, %tb
  %y = sub <4 x i16> %tb, %tb
  %r = call <8 x i16> @llvm.vector.interleave2.v8i16(<4 x i16> %x, <4 x i16> %y)
  ret <8 x i16> %r
}

define <8 x i16> @reversed_members(<8 x i64> %w) {
; CHECK-LABEL: @reversed_members(
; CHECK: %r = call <8 x i16> @llvm.vector.interleave2.v8i16(<4 x i16> %x, <4 x i16> %y)
; CHECK: ret <8 x i16> %r
  %d = call { <4 x i64>, <4 x i64> } @llvm.vector.deinterleave2.v8i64(<8 x i64> %w)
  %a = extractvalue { <4 x i64>, <4 x i64> } %d, 0
  %b = extractvalue { <4 x i64>, <4 x i64> } %d, 1
  %ta = trunc <4 x i64> %a to <4 x i16>
  %tb = trunc <4 x i64> %b to <4 x i16>
  %x = sub <4 x i16> %ta, %tb
  %y = sub <4 x i16> %tb, %ta
  %r = call <8 x i16> @llvm.vector.interleave2.v8i16(<4 x i16> %x, <4 x i16> %y)
  ret <8 x i16> %r
}

declare { <4 x i64>, <4 x i64> } @llvm.vector.deinterleave2.v8i64(<8 x i64>)
declare <8 x i16> @llvm.vector.interleave2.v8i16(<4 x i16>, <4 x i16>)
