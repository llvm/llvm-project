; Test that vectors are scalarized/lowered correctly
; (with both legalization methods).
; RUN: llc -mtriple=i686--  < %s
; RUN: llc -mtriple=i686--  < %s

; A simple test to check copyToParts and copyFromParts.

define <4 x i64> @test_param_0(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   ret <4 x i64> %A
}

define <2 x i32> @test_param_1(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   ret <2 x i32> %B
}

define <4 x i8> @test_param_2(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   ret <4 x i8> %C
}

; Simple tests to check arithmetic and vector operations on types which need to
; be legalized (no loads/stores to/from memory here).

define <4 x i64> @test_arith_0(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   %K = add <4 x i64> %A, <i64 0, i64 1, i64 3, i64 9>
   ret <4 x i64> %K
}

define <2 x i32> @test_arith_1(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   %K = add <2 x i32> %B, <i32 0, i32 1>
   ret <2 x i32> %K
}

define <4 x i8> @test_arith_2(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   %K = add <4 x i8> %C, <i8 0, i8 1, i8 3, i8 9>
   ret <4 x i8> %K
}

define i8 @test_arith_3(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   %K = add <4 x i8> %C, <i8 0, i8 1, i8 3, i8 9>
   %Y = extractelement <4 x i8> %K, i32 1
   ret i8 %Y
}

define <4 x i8> @test_arith_4(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   %Y = insertelement <4 x i8> %C, i8 1, i32 0
   ret <4 x i8> %Y
}

define <4 x i32> @test_arith_5(<4 x i64> %A, <2 x i32> %B, <4 x i32> %C)  {
   %Y = insertelement <4 x i32> %C, i32 1, i32 0
   ret <4 x i32> %Y
}

define <4 x i32> @test_arith_6(<4 x i64> %A, <2 x i32> %B, <4 x i32> %C)  {
   %F = extractelement <2 x i32> %B, i32 1
   %Y = insertelement <4 x i32> %C, i32 %F, i32 0
   ret <4 x i32> %Y
}

define <4 x i64> @test_arith_7(<4 x i64> %A, <2 x i32> %B, <4 x i32> %C)  {
   %F = extractelement <2 x i32> %B, i32 1
   %W = zext i32 %F to i64
   %Y = insertelement <4 x i64> %A, i64 %W, i32 0
   ret <4 x i64> %Y
}

define i64 @test_arith_8(<4 x i64> %A, <2 x i32> %B, <4 x i32> %C)  {
   %F = extractelement <2 x i32> %B, i32 1
   %W = zext i32 %F to i64
   %T = add i64 %W , 11
   ret i64 %T
}

define <4 x i64> @test_arith_9(<4 x i64> %A, <2 x i32> %B, <4 x i16> %C)  {
   %T = add <4 x i16> %C, %C
   %F0 = extractelement <4 x i16> %T, i32 0
   %F1 = extractelement <4 x i16> %T, i32 1
   %W0 = zext i16 %F0 to i64
   %W1 = zext i16 %F1 to i64
   %Y0 = insertelement <4 x i64> %A,  i64 %W0, i32 0
   %Y1 = insertelement <4 x i64> %Y0, i64 %W1, i32 2
   ret <4 x i64> %Y1
}

define <4 x i16> @test_arith_10(<4 x i64> %A, <2 x i32> %B, <4 x i32> %C)  {
   %F = bitcast <2 x i32> %B to <4 x i16>
   %T = add <4 x i16> %F , <i16 0, i16 1, i16 2, i16 3>
   ret <4 x i16> %T
}


; Simple tests to check saving/loading from memory
define <4 x i16> @test_mem_0(<4 x i64> %A, <2 x i32> %B, <4 x i32> %C)  {
   %F = bitcast <2 x i32> %B to <4 x i16>
   %T = add <4 x i16> %F , <i16 0, i16 1, i16 2, i16 3>
   ret <4 x i16> %T
}

