; RUN: opt -S -passes=instcombine %s -o - | FileCheck %s


declare i64 @llrint(double)

; Positive number test
; CHECK-LABEL: define i64 @test_llrint_pos()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 4
define i64 @test_llrint_pos() {
entry:
  %val = call i64 @llrint(double 3.5)
  ret i64 %val
}

; Negative number test
; CHECK-LABEL: define i64 @test_llrint_neg()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 -2
define i64 @test_llrint_neg() {
entry:
  %val = call i64 @llrint(double -2.5)
  ret i64 %val
}

; Zero test
; CHECK-LABEL: define i64 @test_llrint_zero()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 0
define i64 @test_llrint_zero() {
entry:
  %val = call i64 @llrint(double 0.0)
  ret i64 %val
}

; Large value test
; CHECK-LABEL: define i64 @test_llrint_large()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 1000000
define i64 @test_llrint_large() {
entry:
  %val = call i64 @llrint(double 1.0e6)
  ret i64 %val
}

; Rounding test (check ties-to-even)
; CHECK-LABEL: define i64 @test_llrint_round_even()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 2
define i64 @test_llrint_round_even() {
entry:
  %val = call i64 @llrint(double 2.5)
  ret i64 %val
}

; NaN test
; CHECK-LABEL: define i64 @test_llrint_nan()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %val = call i64 @llrint(double 0x7FF8000000000000)
; CHECK-NEXT:    ret i64 %val
define i64 @test_llrint_nan() {
entry:
  %val = call i64 @llrint(double 0x7FF8000000000000) ; NaN
  ret i64 %val
}

; +Inf test
; CHECK-LABEL: define i64 @test_llrint_posinf()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %val = call i64 @llrint(double 0x7FF0000000000000)
; CHECK-NEXT:    ret i64 %val
define i64 @test_llrint_posinf() {
entry:
  %val = call i64 @llrint(double 0x7FF0000000000000) ; +Inf
  ret i64 %val
}

; -Inf test
; CHECK-LABEL: define i64 @test_llrint_neginf()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %val = call i64 @llrint(double 0xFFF0000000000000)
; CHECK-NEXT:    ret i64 %val
define i64 @test_llrint_neginf() {
entry:
  %val = call i64 @llrint(double 0xFFF0000000000000) ; -Inf
  ret i64 %val
}