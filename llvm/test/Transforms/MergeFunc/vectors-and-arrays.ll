; REQUIRES: asserts
; RUN: opt -passes=mergefunc < %s -disable-output -stats | not grep merged
; This used to crash with an assert.

define <2 x i8> @v1(<2 x i8> %x) {
  ret <2 x i8> %x
}

define <4 x i8> @v2(<4 x i8> %x) {
  ret <4 x i8> %x
}

define [2 x i8] @a1([2 x i8] %x) {
  ret [2 x i8] %x
}

define [4 x i8] @a2([4 x i8] %x) {
  ret [4 x i8] %x
}
