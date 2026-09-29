; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s

; Testing that these don't crash/assert. The loop vectorizer can end up
; with odd constructs like this. The code actually generated is incidental.
define <1 x i64> @test_zext(i32 %a) nounwind {
; CHECK-LABEL: test_zext:
  %Cmp = icmp uge i32 %a, 42
  %vec = insertelement <1 x i1> zeroinitializer, i1 %Cmp, i32 0
  %Se = zext <1 x i1> %vec to <1 x i64>
  ret <1 x i64> %Se
}

define <1 x i64> @test_sext(i32 %a) nounwind {
; CHECK-LABEL: test_sext:
  %Cmp = icmp uge i32 %a, 42
  %vec = insertelement <1 x i1> zeroinitializer, i1 %Cmp, i32 0
  %Se = sext <1 x i1> %vec to <1 x i64>
  ret <1 x i64> %Se
}
