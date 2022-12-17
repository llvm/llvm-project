; REQUIRES: asserts
; RUN: opt -passes=inline -mtriple=x86_64-unknown-unknown -S -debug-only=inline-cost < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define i32 @outer1(i32* %ptr, i32 %i) {
  %C = call i32 @inner1(i32* %ptr, i32 %i)
  ret i32 %C
}

; zext from i32 to i64 is free.
; CHECK: Analyzing call of inner1
; CHECK: NumInstructionsSimplified: 3
; CHECK: NumInstructions: 4
define i32 @inner1(i32* %ptr, i32 %i) {
  %E = zext i32 %i to i64
  %G = getelementptr inbounds i32, i32* %ptr, i64 %E
  %L = load i32, i32* %G
  ret i32 %L
}

define i16 @outer2(i8* %ptr) {
  %C = call i16 @inner2(i8* %ptr)
  ret i16 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner2
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i16 @inner2(i8* %ptr) {
  %L = load i8, i8* %ptr
  %E = zext i8 %L to i16
  ret i16 %E
}

define i16 @outer3(i8* %ptr) {
  %C = call i16 @inner3(i8* %ptr)
  ret i16 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner3
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i16 @inner3(i8* %ptr) {
  %L = load i8, i8* %ptr
  %E = sext i8 %L to i16
  ret i16 %E
}

define i32 @outer4(i8* %ptr) {
  %C = call i32 @inner4(i8* %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner4
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner4(i8* %ptr) {
  %L = load i8, i8* %ptr
  %E = zext i8 %L to i32
  ret i32 %E
}

define i32 @outer5(i8* %ptr) {
  %C = call i32 @inner5(i8* %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner5
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner5(i8* %ptr) {
  %L = load i8, i8* %ptr
  %E = sext i8 %L to i32
  ret i32 %E
}

define i32 @outer6(i16* %ptr) {
  %C = call i32 @inner6(i16* %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner6
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner6(i16* %ptr) {
  %L = load i16, i16* %ptr
  %E = zext i16 %L to i32
  ret i32 %E
}

define i32 @outer7(i16* %ptr) {
  %C = call i32 @inner7(i16* %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner7
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner7(i16* %ptr) {
  %L = load i16, i16* %ptr
  %E = sext i16 %L to i32
  ret i32 %E
}

define i64 @outer8(i8* %ptr) {
  %C = call i64 @inner8(i8* %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner8
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner8(i8* %ptr) {
  %L = load i8, i8* %ptr
  %E = zext i8 %L to i64
  ret i64 %E
}

define i64 @outer9(i8* %ptr) {
  %C = call i64 @inner9(i8* %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner9
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner9(i8* %ptr) {
  %L = load i8, i8* %ptr
  %E = sext i8 %L to i64
  ret i64 %E
}

define i64 @outer10(i16* %ptr) {
  %C = call i64 @inner10(i16* %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner10
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner10(i16* %ptr) {
  %L = load i16, i16* %ptr
  %E = zext i16 %L to i64
  ret i64 %E
}

define i64 @outer11(i16* %ptr) {
  %C = call i64 @inner11(i16* %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner11
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner11(i16* %ptr) {
  %L = load i16, i16* %ptr
  %E = sext i16 %L to i64
  ret i64 %E
}

define i64 @outer12(i32* %ptr) {
  %C = call i64 @inner12(i32* %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner12
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner12(i32* %ptr) {
  %L = load i32, i32* %ptr
  %E = zext i32 %L to i64
  ret i64 %E
}

define i64 @outer13(i32* %ptr) {
  %C = call i64 @inner13(i32* %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner13
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner13(i32* %ptr) {
  %L = load i32, i32* %ptr
  %E = sext i32 %L to i64
  ret i64 %E
}
