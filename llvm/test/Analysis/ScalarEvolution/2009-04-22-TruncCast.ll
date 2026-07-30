; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'test1'
; CHECK-NOT: (trunc i{{.*}}ext

define i16 @test1(i8 %x) {
  %A = sext i8 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'test2'
; CHECK-NOT: (trunc i{{.*}}ext

define i8 @test2(i16 %x) {
  %A = sext i16 %x to i32
  %B = trunc i32 %A to i8
  ret i8 %B
}

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'test3'
; CHECK-NOT: (trunc i{{.*}}ext

define i16 @test3(i16 %x) {
  %A = sext i16 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'test4'
; CHECK-NOT: (trunc i{{.*}}ext

define i16 @test4(i8 %x) {
  %A = zext i8 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'test5'
; CHECK-NOT: (trunc i{{.*}}ext

define i8 @test5(i16 %x) {
  %A = zext i16 %x to i32
  %B = trunc i32 %A to i8
  ret i8 %B
}

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'test6'
; CHECK-NOT: (trunc i{{.*}}ext

define i16 @test6(i16 %x) {
  %A = zext i16 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}
