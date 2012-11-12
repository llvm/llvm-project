; Test the basic functionality of integer element promotions of different types.
; This tests checks passing of arguments, loading and storing to memory and
; basic arithmetic.
; RUN: llc -march=x86 < %s
; RUN: llc -march=x86-64 < %s

define <1 x i8> @test_1xi8(<1 x i8> %x, <1 x i8>* %b) {
  %bb = load <1 x i8>* %b
  %tt = xor <1 x i8> %x, %bb
  store <1 x i8> %tt, <1 x i8>* %b
  br label %next

next:
  ret <1 x i8> %tt
}


define <1 x i16> @test_1xi16(<1 x i16> %x, <1 x i16>* %b) {
  %bb = load <1 x i16>* %b
  %tt = xor <1 x i16> %x, %bb
  store <1 x i16> %tt, <1 x i16>* %b
  br label %next

next:
  ret <1 x i16> %tt
}


define <1 x i32> @test_1xi32(<1 x i32> %x, <1 x i32>* %b) {
  %bb = load <1 x i32>* %b
  %tt = xor <1 x i32> %x, %bb
  store <1 x i32> %tt, <1 x i32>* %b
  br label %next

next:
  ret <1 x i32> %tt
}


define <1 x i64> @test_1xi64(<1 x i64> %x, <1 x i64>* %b) {
  %bb = load <1 x i64>* %b
  %tt = xor <1 x i64> %x, %bb
  store <1 x i64> %tt, <1 x i64>* %b
  br label %next

next:
  ret <1 x i64> %tt
}


define <1 x i128> @test_1xi128(<1 x i128> %x, <1 x i128>* %b) {
  %bb = load <1 x i128>* %b
  %tt = xor <1 x i128> %x, %bb
  store <1 x i128> %tt, <1 x i128>* %b
  br label %next

next:
  ret <1 x i128> %tt
}


define <1 x i256> @test_1xi256(<1 x i256> %x, <1 x i256>* %b) {
  %bb = load <1 x i256>* %b
  %tt = xor <1 x i256> %x, %bb
  store <1 x i256> %tt, <1 x i256>* %b
  br label %next

next:
  ret <1 x i256> %tt
}


define <1 x i512> @test_1xi512(<1 x i512> %x, <1 x i512>* %b) {
  %bb = load <1 x i512>* %b
  %tt = xor <1 x i512> %x, %bb
  store <1 x i512> %tt, <1 x i512>* %b
  br label %next

next:
  ret <1 x i512> %tt
}


define <2 x i8> @test_2xi8(<2 x i8> %x, <2 x i8>* %b) {
  %bb = load <2 x i8>* %b
  %tt = xor <2 x i8> %x, %bb
  store <2 x i8> %tt, <2 x i8>* %b
  br label %next

next:
  ret <2 x i8> %tt
}


define <2 x i16> @test_2xi16(<2 x i16> %x, <2 x i16>* %b) {
  %bb = load <2 x i16>* %b
  %tt = xor <2 x i16> %x, %bb
  store <2 x i16> %tt, <2 x i16>* %b
  br label %next

next:
  ret <2 x i16> %tt
}


define <2 x i32> @test_2xi32(<2 x i32> %x, <2 x i32>* %b) {
  %bb = load <2 x i32>* %b
  %tt = xor <2 x i32> %x, %bb
  store <2 x i32> %tt, <2 x i32>* %b
  br label %next

next:
  ret <2 x i32> %tt
}


define <2 x i64> @test_2xi64(<2 x i64> %x, <2 x i64>* %b) {
  %bb = load <2 x i64>* %b
  %tt = xor <2 x i64> %x, %bb
  store <2 x i64> %tt, <2 x i64>* %b
  br label %next

next:
  ret <2 x i64> %tt
}


define <2 x i128> @test_2xi128(<2 x i128> %x, <2 x i128>* %b) {
  %bb = load <2 x i128>* %b
  %tt = xor <2 x i128> %x, %bb
  store <2 x i128> %tt, <2 x i128>* %b
  br label %next

next:
  ret <2 x i128> %tt
}


define <2 x i256> @test_2xi256(<2 x i256> %x, <2 x i256>* %b) {
  %bb = load <2 x i256>* %b
  %tt = xor <2 x i256> %x, %bb
  store <2 x i256> %tt, <2 x i256>* %b
  br label %next

next:
  ret <2 x i256> %tt
}


define <2 x i512> @test_2xi512(<2 x i512> %x, <2 x i512>* %b) {
  %bb = load <2 x i512>* %b
  %tt = xor <2 x i512> %x, %bb
  store <2 x i512> %tt, <2 x i512>* %b
  br label %next

next:
  ret <2 x i512> %tt
}


define <3 x i8> @test_3xi8(<3 x i8> %x, <3 x i8>* %b) {
  %bb = load <3 x i8>* %b
  %tt = xor <3 x i8> %x, %bb
  store <3 x i8> %tt, <3 x i8>* %b
  br label %next

next:
  ret <3 x i8> %tt
}


define <3 x i16> @test_3xi16(<3 x i16> %x, <3 x i16>* %b) {
  %bb = load <3 x i16>* %b
  %tt = xor <3 x i16> %x, %bb
  store <3 x i16> %tt, <3 x i16>* %b
  br label %next

next:
  ret <3 x i16> %tt
}


define <3 x i32> @test_3xi32(<3 x i32> %x, <3 x i32>* %b) {
  %bb = load <3 x i32>* %b
  %tt = xor <3 x i32> %x, %bb
  store <3 x i32> %tt, <3 x i32>* %b
  br label %next

next:
  ret <3 x i32> %tt
}


define <3 x i64> @test_3xi64(<3 x i64> %x, <3 x i64>* %b) {
  %bb = load <3 x i64>* %b
  %tt = xor <3 x i64> %x, %bb
  store <3 x i64> %tt, <3 x i64>* %b
  br label %next

next:
  ret <3 x i64> %tt
}


define <3 x i128> @test_3xi128(<3 x i128> %x, <3 x i128>* %b) {
  %bb = load <3 x i128>* %b
  %tt = xor <3 x i128> %x, %bb
  store <3 x i128> %tt, <3 x i128>* %b
  br label %next

next:
  ret <3 x i128> %tt
}


define <3 x i256> @test_3xi256(<3 x i256> %x, <3 x i256>* %b) {
  %bb = load <3 x i256>* %b
  %tt = xor <3 x i256> %x, %bb
  store <3 x i256> %tt, <3 x i256>* %b
  br label %next

next:
  ret <3 x i256> %tt
}


define <3 x i512> @test_3xi512(<3 x i512> %x, <3 x i512>* %b) {
  %bb = load <3 x i512>* %b
  %tt = xor <3 x i512> %x, %bb
  store <3 x i512> %tt, <3 x i512>* %b
  br label %next

next:
  ret <3 x i512> %tt
}


define <4 x i8> @test_4xi8(<4 x i8> %x, <4 x i8>* %b) {
  %bb = load <4 x i8>* %b
  %tt = xor <4 x i8> %x, %bb
  store <4 x i8> %tt, <4 x i8>* %b
  br label %next

next:
  ret <4 x i8> %tt
}


define <4 x i16> @test_4xi16(<4 x i16> %x, <4 x i16>* %b) {
  %bb = load <4 x i16>* %b
  %tt = xor <4 x i16> %x, %bb
  store <4 x i16> %tt, <4 x i16>* %b
  br label %next

next:
  ret <4 x i16> %tt
}


define <4 x i32> @test_4xi32(<4 x i32> %x, <4 x i32>* %b) {
  %bb = load <4 x i32>* %b
  %tt = xor <4 x i32> %x, %bb
  store <4 x i32> %tt, <4 x i32>* %b
  br label %next

next:
  ret <4 x i32> %tt
}


define <4 x i64> @test_4xi64(<4 x i64> %x, <4 x i64>* %b) {
  %bb = load <4 x i64>* %b
  %tt = xor <4 x i64> %x, %bb
  store <4 x i64> %tt, <4 x i64>* %b
  br label %next

next:
  ret <4 x i64> %tt
}


define <4 x i128> @test_4xi128(<4 x i128> %x, <4 x i128>* %b) {
  %bb = load <4 x i128>* %b
  %tt = xor <4 x i128> %x, %bb
  store <4 x i128> %tt, <4 x i128>* %b
  br label %next

next:
  ret <4 x i128> %tt
}


define <4 x i256> @test_4xi256(<4 x i256> %x, <4 x i256>* %b) {
  %bb = load <4 x i256>* %b
  %tt = xor <4 x i256> %x, %bb
  store <4 x i256> %tt, <4 x i256>* %b
  br label %next

next:
  ret <4 x i256> %tt
}


define <4 x i512> @test_4xi512(<4 x i512> %x, <4 x i512>* %b) {
  %bb = load <4 x i512>* %b
  %tt = xor <4 x i512> %x, %bb
  store <4 x i512> %tt, <4 x i512>* %b
  br label %next

next:
  ret <4 x i512> %tt
}


define <5 x i8> @test_5xi8(<5 x i8> %x, <5 x i8>* %b) {
  %bb = load <5 x i8>* %b
  %tt = xor <5 x i8> %x, %bb
  store <5 x i8> %tt, <5 x i8>* %b
  br label %next

next:
  ret <5 x i8> %tt
}


define <5 x i16> @test_5xi16(<5 x i16> %x, <5 x i16>* %b) {
  %bb = load <5 x i16>* %b
  %tt = xor <5 x i16> %x, %bb
  store <5 x i16> %tt, <5 x i16>* %b
  br label %next

next:
  ret <5 x i16> %tt
}


define <5 x i32> @test_5xi32(<5 x i32> %x, <5 x i32>* %b) {
  %bb = load <5 x i32>* %b
  %tt = xor <5 x i32> %x, %bb
  store <5 x i32> %tt, <5 x i32>* %b
  br label %next

next:
  ret <5 x i32> %tt
}


define <5 x i64> @test_5xi64(<5 x i64> %x, <5 x i64>* %b) {
  %bb = load <5 x i64>* %b
  %tt = xor <5 x i64> %x, %bb
  store <5 x i64> %tt, <5 x i64>* %b
  br label %next

next:
  ret <5 x i64> %tt
}


define <5 x i128> @test_5xi128(<5 x i128> %x, <5 x i128>* %b) {
  %bb = load <5 x i128>* %b
  %tt = xor <5 x i128> %x, %bb
  store <5 x i128> %tt, <5 x i128>* %b
  br label %next

next:
  ret <5 x i128> %tt
}


define <5 x i256> @test_5xi256(<5 x i256> %x, <5 x i256>* %b) {
  %bb = load <5 x i256>* %b
  %tt = xor <5 x i256> %x, %bb
  store <5 x i256> %tt, <5 x i256>* %b
  br label %next

next:
  ret <5 x i256> %tt
}


define <5 x i512> @test_5xi512(<5 x i512> %x, <5 x i512>* %b) {
  %bb = load <5 x i512>* %b
  %tt = xor <5 x i512> %x, %bb
  store <5 x i512> %tt, <5 x i512>* %b
  br label %next

next:
  ret <5 x i512> %tt
}


