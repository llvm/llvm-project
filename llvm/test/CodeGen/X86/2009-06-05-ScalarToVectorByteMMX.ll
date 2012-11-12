; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu -mattr=+mmx,+sse2 | not grep movl

define <8 x i8> @a(i8 zeroext %x) nounwind {
  %r = insertelement <8 x i8> undef, i8 %x, i32 0
  ret <8 x i8> %r
}

