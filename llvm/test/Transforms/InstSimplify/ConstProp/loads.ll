; RUN: opt < %s -data-layout="e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64" -instcombine -S | FileCheck %s --check-prefix=LE
; RUN: opt < %s -data-layout="E-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64" -instcombine -S | FileCheck %s --check-prefix=BE

; {{ 0xDEADBEEF, 0xBA }, 0xCAFEBABE}
@g1 = constant {{i32,i8},i32} {{i32,i8} { i32 -559038737, i8 186 }, i32 -889275714 }
@g2 = constant double 1.0
; { 0x7B, 0x06B1BFF8 }
@g3 = constant {i64, i64} { i64 123, i64 112312312 }

; Simple load
define i32 @test1() {
  %r = load i32, i32* getelementptr ({{i32,i8},i32}, {{i32,i8},i32}* @g1, i32 0, i32 0, i32 0)
  ret i32 %r

; 0xDEADBEEF
; LE-LABEL: @test1(
; LE: ret i32 -559038737

; 0xDEADBEEF
; BE-LABEL: @test1(
; BE: ret i32 -559038737
}

; PR3152
; Load of first 16 bits of 32-bit value.
define i16 @test2() {
  %r = load i16, i16* bitcast(i32* getelementptr ({{i32,i8},i32}, {{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*)
  ret i16 %r

; 0xBEEF
; LE-LABEL: @test2(
; LE: ret i16 -16657

; 0xDEAD
; BE-LABEL: @test2(
; BE: ret i16 -8531
}

define i16 @test2_addrspacecast() {
  %r = load i16, i16 addrspace(1)* addrspacecast(i32* getelementptr ({{i32,i8},i32}, {{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16 addrspace(1)*)
  ret i16 %r

; FIXME: Should be able to load through a constant addrspacecast.
; 0xBEEF
; LE-LABEL: @test2_addrspacecast(
; XLE: ret i16 -16657
; LE: load i16, i16 addrspace(1)* addrspacecast

; 0xDEAD
; BE-LABEL: @test2_addrspacecast(
; XBE: ret i16 -8531
; BE: load i16, i16 addrspace(1)* addrspacecast
}

; Load of second 16 bits of 32-bit value.
define i16 @test3() {
  %r = load i16, i16* getelementptr(i16, i16* bitcast(i32* getelementptr ({{i32,i8},i32}, {{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*), i32 1)
  ret i16 %r

; 0xDEAD
; LE-LABEL: @test3(
; LE: ret i16 -8531

; 0xBEEF
; BE-LABEL: @test3(
; BE: ret i16 -16657
}

; Load of 8 bit field + tail padding.
define i16 @test4() {
  %r = load i16, i16* getelementptr(i16, i16* bitcast(i32* getelementptr ({{i32,i8},i32}, {{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*), i32 2)
  ret i16 %r

; 0x00BA
; LE-LABEL: @test4(
; LE: ret i16 186

; 0xBA00
; BE-LABEL: @test4(
; BE: ret i16 -17920
}

; Load of double bits.
define i64 @test6() {
  %r = load i64, i64* bitcast(double* @g2 to i64*)
  ret i64 %r

; 0x3FF_0000000000000
; LE-LABEL: @test6(
; LE: ret i64 4607182418800017408

; 0x3FF_0000000000000
; BE-LABEL: @test6(
; BE: ret i64 4607182418800017408
}

; Load of double bits.
define i16 @test7() {
  %r = load i16, i16* bitcast(double* @g2 to i16*)
  ret i16 %r

; 0x0000
; LE-LABEL: @test7(
; LE: ret i16 0

; 0x3FF0
; BE-LABEL: @test7(
; BE: ret i16 16368
}

; Double load.
define double @test8() {
  %r = load double, double* bitcast({{i32,i8},i32}* @g1 to double*)
  ret double %r

; LE-LABEL: @test8(
; LE: ret double 0xBADEADBEEF

; BE-LABEL: @test8(
; BE: ret double 0xDEADBEEFBA000000
}


; i128 load.
define i128 @test9() {
  %r = load i128, i128* bitcast({i64, i64}* @g3 to i128*)
  ret i128 %r

; 0x00000000_06B1BFF8_00000000_0000007B
; LE-LABEL: @test9(
; LE: ret i128 2071796475790618158476296315

; 0x00000000_0000007B_00000000_06B1BFF8
; BE-LABEL: @test9(
; BE: ret i128 2268949521066387161080
}

; vector load.
define <2 x i64> @test10() {
  %r = load <2 x i64>, <2 x i64>* bitcast({i64, i64}* @g3 to <2 x i64>*)
  ret <2 x i64> %r

; LE-LABEL: @test10(
; LE: ret <2 x i64> <i64 123, i64 112312312>

; BE-LABEL: @test10(
; BE: ret <2 x i64> <i64 123, i64 112312312>
}


; PR5287
; { 0xA1, 0x08 }
@g4 = internal constant { i8, i8 } { i8 -95, i8 8 }

define i16 @test11() nounwind {
entry:
  %a = load i16, i16* bitcast ({ i8, i8 }* @g4 to i16*)
  ret i16 %a

; 0x08A1
; LE-LABEL: @test11(
; LE: ret i16 2209

; 0xA108
; BE-LABEL: @test11(
; BE: ret i16 -24312
}


; PR5551
@test12g = private constant [6 x i8] c"a\00b\00\00\00"

define i16 @test12() {
  %a = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* bitcast ([6 x i8]* @test12g to [3 x i16]*), i32 0, i64 1)
  ret i16 %a

; 0x0062
; LE-LABEL: @test12(
; LE: ret i16 98

; 0x6200
; BE-LABEL: @test12(
; BE: ret i16 25088
}


; PR5978
@g5 = constant i8 4
define i1 @test13() {
  %A = load i1, i1* bitcast (i8* @g5 to i1*)
  ret i1 %A

; LE-LABEL: @test13(
; LE: ret i1 false

; BE-LABEL: @test13(
; BE: ret i1 false
}

@g6 = constant [2 x i8*] [i8* inttoptr (i64 1 to i8*), i8* inttoptr (i64 2 to i8*)]
define i64 @test14() nounwind {
entry:
  %tmp = load i64, i64* bitcast ([2 x i8*]* @g6 to i64*)
  ret i64 %tmp

; LE-LABEL: @test14(
; LE: ret i64 1

; BE-LABEL: @test14(
; BE: ret i64 1
}

; Check with address space pointers
@g6_as1 = constant [2 x i8 addrspace(1)*] [i8 addrspace(1)* inttoptr (i16 1 to i8 addrspace(1)*), i8 addrspace(1)* inttoptr (i16 2 to i8 addrspace(1)*)]
define i16 @test14_as1() nounwind {
entry:
  %tmp = load i16, i16* bitcast ([2 x i8 addrspace(1)*]* @g6_as1 to i16*)
  ret i16 %tmp

; LE: @test14_as1
; LE: ret i16 1

; BE: @test14_as1
; BE: ret i16 1
}

define i64 @test15() nounwind {
entry:
  %tmp = load i64, i64* bitcast (i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @g6, i32 0, i64 1) to i64*)
  ret i64 %tmp

; LE-LABEL: @test15(
; LE: ret i64 2

; BE-LABEL: @test15(
; BE: ret i64 2
}

@gv7 = constant [4 x i8*] [i8* null, i8* inttoptr (i64 -14 to i8*), i8* null, i8* null]
define i64 @test16.1() {
  %v = load i64, i64* bitcast ([4 x i8*]* @gv7 to i64*), align 8
  ret i64 %v

; LE-LABEL: @test16.1(
; LE: ret i64 0

; BE-LABEL: @test16.1(
; BE: ret i64 0
}

define i64 @test16.2() {
  %v = load i64, i64* bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @gv7, i64 0, i64 1) to i64*), align 8
  ret i64 %v

; LE-LABEL: @test16.2(
; LE: ret i64 -14

; BE-LABEL: @test16.2(
; BE: ret i64 -14
}

define i64 @test16.3() {
  %v = load i64, i64* bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @gv7, i64 0, i64 2) to i64*), align 8
  ret i64 %v

; LE-LABEL: @test16.3(
; LE: ret i64 0

; BE-LABEL: @test16.3(
; BE: ret i64 0
}

@g7 = constant {[0 x i32], [0 x i8], {}*} { [0 x i32] undef, [0 x i8] undef, {}* null }

define i64* @test_leading_zero_size_elems() {
  %v = load i64*, i64** bitcast ({[0 x i32], [0 x i8], {}*}* @g7 to i64**)
  ret i64* %v

; LE-LABEL: @test_leading_zero_size_elems(
; LE: ret i64* null

; BE-LABEL: @test_leading_zero_size_elems(
; BE: ret i64* null
}

@g8 = constant {[4294967295 x [0 x i32]], i64} { [4294967295 x [0 x i32]] undef, i64 123 }

define i64 @test_leading_zero_size_elems_big() {
  %v = load i64, i64* bitcast ({[4294967295 x [0 x i32]], i64}* @g8 to i64*)
  ret i64 %v

; LE-LABEL: @test_leading_zero_size_elems_big(
; LE: ret i64 123

; BE-LABEL: @test_leading_zero_size_elems_big(
; BE: ret i64 123
}

@g9 = constant [4294967295 x [0 x i32]] zeroinitializer

define i64 @test_array_of_zero_size_array() {
  %v = load i64, i64* bitcast ([4294967295 x [0 x i32]]* @g9 to i64*)
  ret i64 %v

; LE-LABEL: @test_array_of_zero_size_array(
; LE: ret i64 0

; BE-LABEL: @test_array_of_zero_size_array(
; BE: ret i64 0
}
