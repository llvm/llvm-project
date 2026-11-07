; RUN: llc --mtriple=loongarch64 --mattr=+lsx %s -o - | FileCheck %s
; RUN: llc --mtriple=loongarch32 --mattr=+32s,+lsx %s -o - | FileCheck %s
; RUN: llc --mtriple=loongarch64 --mattr=+lasx %s -o - | FileCheck %s

; The operand of a vector sign extension may still have an illegal type when
; its result type is split during type legalization: results are legalized
; before operands, so a <8 x i2> value has not been promoted yet when
; <8 x i32> is split.  The LSX custom lowering must leave such nodes to the
; generic legalizer, otherwise it asks for an MVT that does not exist.
;
; These tests only check that we no longer crash; the generated code is
; expected to improve once the source vector is widened first.
;
; The same path is reached from the loop vectorizer for _BitInt(24)
; (issue #222088): the promoted v4i24 setcc result is sign-extended to v4i64.

define <8 x i32> @shl_sext_8i2_to_8i32(<8 x i2> %x) {
; CHECK-LABEL: shl_sext_8i2_to_8i32:
; CHECK:       ret
  %sh = shl <8 x i2> %x, %x
  %r = sext <8 x i2> %sh to <8 x i32>
  ret <8 x i32> %r
}

; The producer of the illegal value is irrelevant: any opcode that leaves the
; operand illegal at this point triggers the same path.
define <8 x i32> @add_sext_8i2_to_8i32(<8 x i2> %x) {
; CHECK-LABEL: add_sext_8i2_to_8i32:
; CHECK:       ret
  %add = add <8 x i2> %x, %x
  %r = sext <8 x i2> %add to <8 x i32>
  ret <8 x i32> %r
}

define <8 x i32> @shl_sext_8i3_to_8i32(<8 x i3> %x) {
; CHECK-LABEL: shl_sext_8i3_to_8i32:
; CHECK:       ret
  %sh = shl <8 x i3> %x, %x
  %r = sext <8 x i3> %sh to <8 x i32>
  ret <8 x i32> %r
}

define <4 x i64> @shl_sext_4i2_to_4i64(<4 x i2> %x) {
; CHECK-LABEL: shl_sext_4i2_to_4i64:
; CHECK:       ret
  %sh = shl <4 x i2> %x, %x
  %r = sext <4 x i2> %sh to <4 x i64>
  ret <4 x i64> %r
}

define <16 x i16> @shl_sext_16i2_to_16i16(<16 x i2> %x) {
; CHECK-LABEL: shl_sext_16i2_to_16i16:
; CHECK:       ret
  %sh = shl <16 x i2> %x, %x
  %r = sext <16 x i2> %sh to <16 x i16>
  ret <16 x i16> %r
}

define <4 x i64> @select_i24_to_i64(<4 x i24> %a, <4 x i24> %b) {
; CHECK-LABEL: select_i24_to_i64:
; CHECK:       ret
  %cmp = icmp slt <4 x i24> %a, %b
  %r = select <4 x i1> %cmp, <4 x i64> splat (i64 5), <4 x i64> zeroinitializer
  ret <4 x i64> %r
}
