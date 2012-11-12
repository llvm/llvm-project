; RUN: llc < %s -march=x86-64
; PR 9267

define<4 x i32> @func_16_32() {
  %F = load <4 x i16>* undef
  %G = zext <4 x i16> %F to <4 x i32>
  %H = load <4 x i16>* undef
  %Y = zext <4 x i16> %H to <4 x i32>
  %T = add <4 x i32> %Y, %G
  store <4 x i32>%T , <4 x i32>* undef
  ret <4 x i32> %T
}

define<4 x i64> @func_16_64() {
  %F = load <4 x i16>* undef
  %G = zext <4 x i16> %F to <4 x i64>
  %H = load <4 x i16>* undef
  %Y = zext <4 x i16> %H to <4 x i64>
  %T = xor <4 x i64> %Y, %G
  store <4 x i64>%T , <4 x i64>* undef
  ret <4 x i64> %T
}

define<4 x i64> @func_32_64() {
  %F = load <4 x i32>* undef
  %G = zext <4 x i32> %F to <4 x i64>
  %H = load <4 x i32>* undef
  %Y = zext <4 x i32> %H to <4 x i64>
  %T = or <4 x i64> %Y, %G
  ret <4 x i64> %T
}

define<4 x i16> @func_8_16() {
  %F = load <4 x i8>* undef
  %G = zext <4 x i8> %F to <4 x i16>
  %H = load <4 x i8>* undef
  %Y = zext <4 x i8> %H to <4 x i16>
  %T = add <4 x i16> %Y, %G
  ret <4 x i16> %T
}

define<4 x i32> @func_8_32() {
  %F = load <4 x i8>* undef
  %G = zext <4 x i8> %F to <4 x i32>
  %H = load <4 x i8>* undef
  %Y = zext <4 x i8> %H to <4 x i32>
  %T = sub <4 x i32> %Y, %G
  ret <4 x i32> %T
}

define<4 x i64> @func_8_64() {
  %F = load <4 x i8>* undef
  %G = zext <4 x i8> %F to <4 x i64>
  %H = load <4 x i8>* undef
  %Y = zext <4 x i8> %H to <4 x i64>
  %T = add <4 x i64> %Y, %G
  ret <4 x i64> %T
}

define<4 x i32> @const_16_32() {
  %G = zext <4 x i16> <i16 0, i16 3, i16 8, i16 7> to <4 x i32>
  ret <4 x i32> %G
}

define<4 x i64> @const_16_64() {
  %G = zext <4 x i16> <i16 0, i16 3, i16 8, i16 7> to <4 x i64>
  ret <4 x i64> %G
}

