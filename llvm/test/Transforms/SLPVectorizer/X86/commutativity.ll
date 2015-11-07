; RUN: opt -slp-vectorizer < %s -S | FileCheck %s

; Verify that the SLP vectorizer is able to figure out that commutativity
; offers the possibility to splat/broadcast %c and thus make it profitable
; to vectorize this case


; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@cle = external unnamed_addr global [32 x i8], align 16
@cle32 = external unnamed_addr global [32 x i32], align 16


; Check that we correctly detect a splat/broadcast by leveraging the
; commutativity property of `xor`.

; CHECK-LABEL:  @splat
; CHECK:  store <16 x i8>
define void @splat(i8 %a, i8 %b, i8 %c) {
  %1 = xor i8 %c, %a
  store i8 %1, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 0), align 16
  %2 = xor i8 %a, %c
  store i8 %2, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 1)
  %3 = xor i8 %a, %c
  store i8 %3, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 2)
  %4 = xor i8 %a, %c
  store i8 %4, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 3)
  %5 = xor i8 %c, %a
  store i8 %5, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 4)
  %6 = xor i8 %c, %b
  store i8 %6, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 5)
  %7 = xor i8 %c, %a
  store i8 %7, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 6)
  %8 = xor i8 %c, %b
  store i8 %8, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 7)
  %9 = xor i8 %a, %c
  store i8 %9, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 8)
  %10 = xor i8 %a, %c
  store i8 %10, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 9)
  %11 = xor i8 %a, %c
  store i8 %11, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 10)
  %12 = xor i8 %a, %c
  store i8 %12, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 11)
  %13 = xor i8 %a, %c
  store i8 %13, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 12)
  %14 = xor i8 %a, %c
  store i8 %14, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 13)
  %15 = xor i8 %a, %c
  store i8 %15, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 14)
  %16 = xor i8 %a, %c
  store i8 %16, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @cle, i64 0, i64 15)
  ret void
}



; Check that we correctly detect that we can have the same opcode on one side by
; leveraging the commutativity property of `xor`.

; CHECK-LABEL:  @same_opcode_on_one_side
; CHECK:  store <4 x i32>
define void @same_opcode_on_one_side(i32 %a, i32 %b, i32 %c) {
  %add1 = add i32 %c, %a
  %add2 = add i32 %c, %a
  %add3 = add i32 %a, %c
  %add4 = add i32 %c, %a
  %1 = xor i32 %add1, %a
  store i32 %1, i32* getelementptr inbounds ([32 x i32], [32 x i32]* @cle32, i64 0, i64 0), align 16
  %2 = xor i32 %b, %add2
  store i32 %2, i32* getelementptr inbounds ([32 x i32], [32 x i32]* @cle32, i64 0, i64 1)
  %3 = xor i32 %c, %add3
  store i32 %3, i32* getelementptr inbounds ([32 x i32], [32 x i32]* @cle32, i64 0, i64 2)
  %4 = xor i32 %a, %add4
  store i32 %4, i32* getelementptr inbounds ([32 x i32], [32 x i32]* @cle32, i64 0, i64 3)
  ret void
}
