; RUN: llc < %s -mtriple=x86_64--

define<4 x i8> @func_8_64() {
  %F = load <4 x i64>, ptr undef
  %G = trunc <4 x i64> %F to <4 x i8>
  %H = load <4 x i64>, ptr undef
  %Y = trunc <4 x i64> %H to <4 x i8>
  %T = add <4 x i8> %Y, %G
  ret <4 x i8> %T
}

