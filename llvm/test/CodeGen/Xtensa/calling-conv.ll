; RUN: llc -mtriple=xtensa -O1 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=XTENSA

; Check placement of first 6 arguments in registers and 7th argument on stack
define dso_local i32 @test1(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, ptr nocapture noundef readonly byval(i32) align 4 %6) {
; XTENSA-LABEL: @test1
; XTENSA:  add	 a8, a7, a2
; XTENSA:  l32i a9, a1, 0
; XTENSA:  add  a2, a8, a9
; XTENSA:  ret
  %8 = load i32, ptr %6, align 4
  %9 = add nsw i32 %5, %0
  %10 = add nsw i32 %9, %8
  ret i32 %10
}

; Check placement of second i64 argument in registers
define dso_local i32 @test2(i32 noundef %0, i64 noundef %1, i32 noundef %2) {
; XTENSA-LABEL: @test2
; XTENSA:  add	 a8, a6, a2
; XTENSA:  add	 a2, a8, a4
; XTENSA:  ret
  %4 = trunc i64 %1 to i32
  %5 = add nsw i32 %2, %0
  %6 = add nsw i32 %5, %4
  ret i32 %6
}

; Check placement of first argument typeof i8 in register
define dso_local i32 @test3(i8 noundef signext %0, i64 noundef %1, i32 noundef %2) {
; XTENSA-LABEL: @test3
; XTENSA:  add  a8, a2, a6
; XTENSA:  add  a2, a8, a4
; XTENSA:  ret
  %4 = trunc i64 %1 to i32
  %5 = sext i8 %0 to i32
  %6 = add nsw i32 %5, %2
  %7 = add nsw i32 %6, %4
  ret i32 %7
}

; Check placement of 4th argument typeof i64 on stack
define dso_local i32 @test4(i8 noundef signext %0, i64 noundef %1, i32 noundef %2, ptr nocapture noundef readonly byval(i64) align 8 %3) {
; XTENSA-LABEL: @test4
; XTENSA: add  a8, a2, a6
; XTENSA: add  a8, a8, a4
; XTENSA: l32i a9, a1, 0
; XTENSA: add  a2, a8, a9
; XTENSA: ret
  %5 = load i64, ptr %3, align 8
  %6 = trunc i64 %1 to i32
  %7 = trunc i64 %5 to i32
  %8 = sext i8 %0 to i32
  %9 = add nsw i32 %8, %2
  %10 = add nsw i32 %9, %6
  %11 = add nsw i32 %10, %7
  ret i32 %11
}

; Check placement of 128 bit structure on registers
define dso_local i32 @test5([4 x i32] %0, i32 noundef %1) {
; XTENSA-LABEL: @test5
; XTENSA: add  a2, a2, a6
; XTENSA: ret
  %3 = extractvalue [4 x i32] %0, 0
  %4 = add nsw i32 %3, %1
  ret i32 %4
}

; Check placement of 128 bit structure on stack
define dso_local i32 @test6(i32 noundef %0, [4 x i32] %1) {
; XTENSA-LABEL: @test6
; XTENSA: add a2, a3, a2
; XTENSA: ret
  %3 = extractvalue [4 x i32] %1, 0
  %4 = add nsw i32 %3, %0
  ret i32 %4
}
