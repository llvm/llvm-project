; RUN: opt -constprop < %s

; Make sure we don't crash on this one

define <8 x i8> @test_truc_vec() {
  %x = bitcast <2 x i64> <i64 1, i64 2> to <8 x i16>
  %y = trunc <8 x i16> %x to <8 x i8>
  ret <8 x i8> %y
}
