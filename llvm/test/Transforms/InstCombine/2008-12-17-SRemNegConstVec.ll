; RUN: opt < %s -instcombine -S | grep "i8 2, i8 2"
; PR2756

define <2 x i8> @foo(<2 x i8> %x) {
  %A = srem <2 x i8> %x, <i8 2, i8 -2>
  ret <2 x i8> %A
}
