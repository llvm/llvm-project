; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=-sse4.1

target triple = "x86_64-unknown-linux-gnu"

define void @m387(ptr %p, ptr %q) {
  %t = load <2 x i8>, ptr %p
  %r = sext <2 x i8> %t to <2 x i16>
  store <2 x i16> %r, ptr %q
  ret void
}

