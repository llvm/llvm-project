; RUN: llc -mtriple=i686-- < %s
; PR13220

define <8 x i32> @foo(<8 x i96> %x) {
  %a = lshr <8 x i96> %x, <i96 1, i96 1, i96 1, i96 1, i96 1, i96 1, i96 1, i96 1>
  %b = trunc <8 x i96> %a to <8 x i32>
  ret <8 x i32> %b
}

define <8 x i32> @bar(<8 x i97> %x) {
  %a = lshr <8 x i97> %x, <i97 1, i97 1, i97 1, i97 1, i97 1, i97 1, i97 1, i97 1>
  %b = trunc <8 x i97> %a to <8 x i32>
  ret <8 x i32> %b
}

define <8 x i32> @bax() {
  %a = lshr <8 x i96> <i96 4, i96 4, i96 4, i96 4, i96 4, i96 4, i96 4, i96 4>, <i96 1, i96 1, i96 1, i96 1, i96 1, i96 1, i96 1, i96 1>
  %b = trunc <8 x i96> %a to <8 x i32>
  ret <8 x i32> %b
}
