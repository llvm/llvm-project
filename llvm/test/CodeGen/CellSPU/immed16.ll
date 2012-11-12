; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep "ilh" %t1.s | count 11
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i16 @test_1() {
  %x = alloca i16, align 16
  store i16 419, i16* %x        ;; ILH via pattern
  ret i16 0
}

define i16 @test_2() {
  %x = alloca i16, align 16
  store i16 1023, i16* %x       ;; ILH via pattern
  ret i16 0
}

define i16 @test_3() {
  %x = alloca i16, align 16
  store i16 -1023, i16* %x      ;; ILH via pattern
  ret i16 0
}

define i16 @test_4() {
  %x = alloca i16, align 16
  store i16 32767, i16* %x      ;; ILH via pattern
  ret i16 0
}

define i16 @test_5() {
  %x = alloca i16, align 16
  store i16 -32768, i16* %x     ;; ILH via pattern
  ret i16 0
}

define i16 @test_6() {
  ret i16 0
}


