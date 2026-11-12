; REQUIRES: asserts
; RUN: opt -passes=inline -S -debug-only=inline-cost < %s 2>&1 | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-ibm-linux-gnu"

define i16 @outer1(ptr %ptr) {
  %C = call i16 @inner1(ptr %ptr)
  ret i16 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner1
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i16 @inner1(ptr %ptr) {
  %L = load i8, ptr %ptr
  %E = zext i8 %L to i16
  ret i16 %E
}

define i32 @outer2(ptr %ptr) {
  %C = call i32 @inner2(ptr %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner2
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner2(ptr %ptr) {
  %L = load i8, ptr %ptr
  %E = zext i8 %L to i32
  ret i32 %E
}

define i32 @outer3(ptr %ptr) {
  %C = call i32 @inner3(ptr %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner3
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner3(ptr %ptr) {
  %L = load i16, ptr %ptr
  %E = zext i16 %L to i32
  ret i32 %E
}

define i32 @outer4(ptr %ptr) {
  %C = call i32 @inner4(ptr %ptr)
  ret i32 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner4
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i32 @inner4(ptr %ptr) {
  %L = load i16, ptr %ptr
  %E = sext i16 %L to i32
  ret i32 %E
}

define i64 @outer5(ptr %ptr) {
  %C = call i64 @inner5(ptr %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner5
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner5(ptr %ptr) {
  %L = load i8, ptr %ptr
  %E = zext i8 %L to i64
  ret i64 %E
}

define i64 @outer6(ptr %ptr) {
  %C = call i64 @inner6(ptr %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner6
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner6(ptr %ptr) {
  %L = load i16, ptr %ptr
  %E = zext i16 %L to i64
  ret i64 %E
}

define i64 @outer7(ptr %ptr) {
  %C = call i64 @inner7(ptr %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner7
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner7(ptr %ptr) {
  %L = load i16, ptr %ptr
  %E = sext i16 %L to i64
  ret i64 %E
}

define i64 @outer8(ptr %ptr) {
  %C = call i64 @inner8(ptr %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner8
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner8(ptr %ptr) {
  %L = load i32, ptr %ptr
  %E = zext i32 %L to i64
  ret i64 %E
}

define i64 @outer9(ptr %ptr) {
  %C = call i64 @inner9(ptr %ptr)
  ret i64 %C
}

; It is an ExtLoad.
; CHECK: Analyzing call of inner9
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 3
define i64 @inner9(ptr %ptr) {
  %L = load i32, ptr %ptr
  %E = sext i32 %L to i64
  ret i64 %E
}
