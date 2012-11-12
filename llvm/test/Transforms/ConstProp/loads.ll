; RUN: opt < %s -default-data-layout="e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64" -instcombine -S | FileCheck %s --check-prefix=LE
; RUN: opt < %s -default-data-layout="E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64" -instcombine -S | FileCheck %s --check-prefix=BE

; {{ 0xDEADBEEF, 0xBA }, 0xCAFEBABE}
@g1 = constant {{i32,i8},i32} {{i32,i8} { i32 -559038737, i8 186 }, i32 -889275714 }
@g2 = constant double 1.0
; { 0x7B, 0x06B1BFF8 }
@g3 = constant {i64, i64} { i64 123, i64 112312312 }

; Simple load
define i32 @test1() {
  %r = load i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0)
  ret i32 %r

; 0xDEADBEEF
; LE: @test1
; LE: ret i32 -559038737

; 0xDEADBEEF
; BE: @test1
; BE: ret i32 -559038737
}

; PR3152
; Load of first 16 bits of 32-bit value.
define i16 @test2() {
  %r = load i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*)
  ret i16 %r

; 0xBEEF
; LE: @test2
; LE: ret i16 -16657

; 0xDEAD
; BE: @test2
; BE: ret i16 -8531
}

; Load of second 16 bits of 32-bit value.
define i16 @test3() {
  %r = load i16* getelementptr(i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*), i32 1)
  ret i16 %r

; 0xDEAD
; LE: @test3
; LE: ret i16 -8531

; 0xBEEF
; BE: @test3
; BE: ret i16 -16657
}

; Load of 8 bit field + tail padding.
define i16 @test4() {
  %r = load i16* getelementptr(i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*), i32 2)
  ret i16 %r

; 0x00BA
; LE: @test4
; LE: ret i16 186

; 0xBA00
; BE: @test4
; BE: ret i16 -17920
}

; Load of double bits.
define i64 @test6() {
  %r = load i64* bitcast(double* @g2 to i64*)
  ret i64 %r

; 0x3FF_0000000000000
; LE: @test6
; LE: ret i64 4607182418800017408

; 0x3FF_0000000000000
; BE: @test6
; BE: ret i64 4607182418800017408
}

; Load of double bits.
define i16 @test7() {
  %r = load i16* bitcast(double* @g2 to i16*)
  ret i16 %r

; 0x0000
; LE: @test7
; LE: ret i16 0

; 0x3FF0
; BE: @test7
; BE: ret i16 16368
}

; Double load.
define double @test8() {
  %r = load double* bitcast({{i32,i8},i32}* @g1 to double*)
  ret double %r

; LE: @test8
; LE: ret double 0xBADEADBEEF

; BE: @test8
; BE: ret double 0xDEADBEEFBA000000
}


; i128 load.
define i128 @test9() {
  %r = load i128* bitcast({i64, i64}* @g3 to i128*)
  ret i128 %r

; 0x00000000_06B1BFF8_00000000_0000007B
; LE: @test9
; LE: ret i128 2071796475790618158476296315

; 0x00000000_0000007B_00000000_06B1BFF8
; BE: @test9
; BE: ret i128 2268949521066387161080
}

; vector load.
define <2 x i64> @test10() {
  %r = load <2 x i64>* bitcast({i64, i64}* @g3 to <2 x i64>*)
  ret <2 x i64> %r

; LE: @test10
; LE: ret <2 x i64> <i64 123, i64 112312312>

; BE: @test10
; BE: ret <2 x i64> <i64 123, i64 112312312>
}


; PR5287
; { 0xA1, 0x08 }
@g4 = internal constant { i8, i8 } { i8 -95, i8 8 }

define i16 @test11() nounwind {
entry:
  %a = load i16* bitcast ({ i8, i8 }* @g4 to i16*)
  ret i16 %a

; 0x08A1
; LE: @test11
; LE: ret i16 2209

; 0xA108
; BE: @test11
; BE: ret i16 -24312
}


; PR5551
@test12g = private constant [6 x i8] c"a\00b\00\00\00"

define i16 @test12() {
  %a = load i16* getelementptr inbounds ([3 x i16]* bitcast ([6 x i8]* @test12g to [3 x i16]*), i32 0, i64 1) 
  ret i16 %a

; 0x0062
; LE: @test12
; LE: ret i16 98

; 0x6200
; BE: @test12
; BE: ret i16 25088
}


; PR5978
@g5 = constant i8 4
define i1 @test13() {
  %A = load i1* bitcast (i8* @g5 to i1*)
  ret i1 %A

; LE: @test13
; LE: ret i1 false

; BE: @test13
; BE: ret i1 false
}

@g6 = constant [2 x i8*] [i8* inttoptr (i64 1 to i8*), i8* inttoptr (i64 2 to i8*)]
define i64 @test14() nounwind {
entry:
  %tmp = load i64* bitcast ([2 x i8*]* @g6 to i64*)
  ret i64 %tmp

; LE: @test14
; LE: ret i64 1

; BE: @test14
; BE: ret i64 1
}

define i64 @test15() nounwind {
entry:
  %tmp = load i64* bitcast (i8** getelementptr inbounds ([2 x i8*]* @g6, i32 0, i64 1) to i64*)
  ret i64 %tmp

; LE: @test15
; LE: ret i64 2

; BE: @test15
; BE: ret i64 2
}
