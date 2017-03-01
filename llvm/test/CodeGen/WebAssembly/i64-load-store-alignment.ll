; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s

; Test loads and stores with custom alignment values.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: ldi64_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a1(i64 *%p) {
  %v = load i64, i64* %p, align 1
  ret i64 %v
}

; CHECK-LABEL: ldi64_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a2(i64 *%p) {
  %v = load i64, i64* %p, align 2
  ret i64 %v
}

; CHECK-LABEL: ldi64_a4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a4(i64 *%p) {
  %v = load i64, i64* %p, align 4
  ret i64 %v
}

; 8 is the default alignment for i64 so no attribute is needed.

; CHECK-LABEL: ldi64_a8:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a8(i64 *%p) {
  %v = load i64, i64* %p, align 8
  ret i64 %v
}

; The default alignment in LLVM is the same as the defualt alignment in wasm.

; CHECK-LABEL: ldi64:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64(i64 *%p) {
  %v = load i64, i64* %p
  ret i64 %v
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: ldi64_a16:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a16(i64 *%p) {
  %v = load i64, i64* %p, align 16
  ret i64 %v
}

; Extending loads.

; CHECK-LABEL: ldi8_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi8_a1(i8 *%p) {
  %v = load i8, i8* %p, align 1
  %w = zext i8 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi8_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi8_a2(i8 *%p) {
  %v = load i8, i8* %p, align 2
  %w = zext i8 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi16_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load16_u $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi16_a1(i16 *%p) {
  %v = load i16, i16* %p, align 1
  %w = zext i16 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi16_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load16_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi16_a2(i16 *%p) {
  %v = load i16, i16* %p, align 2
  %w = zext i16 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi16_a4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load16_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi16_a4(i16 *%p) {
  %v = load i16, i16* %p, align 4
  %w = zext i16 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a1(i32 *%p) {
  %v = load i32, i32* %p, align 1
  %w = zext i32 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a2(i32 *%p) {
  %v = load i32, i32* %p, align 2
  %w = zext i32 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a4(i32 *%p) {
  %v = load i32, i32* %p, align 4
  %w = zext i32 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a8:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a8(i32 *%p) {
  %v = load i32, i32* %p, align 8
  %w = zext i32 %v to i64
  ret i64 %w
}

; Stores.

; CHECK-LABEL: sti64_a1:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a1(i64 *%p, i64 %v) {
  store i64 %v, i64* %p, align 1
  ret void
}

; CHECK-LABEL: sti64_a2:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store 0($0):p2align=1, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a2(i64 *%p, i64 %v) {
  store i64 %v, i64* %p, align 2
  ret void
}

; CHECK-LABEL: sti64_a4:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a4(i64 *%p, i64 %v) {
  store i64 %v, i64* %p, align 4
  ret void
}

; 8 is the default alignment for i32 so no attribute is needed.

; CHECK-LABEL: sti64_a8:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a8(i64 *%p, i64 %v) {
  store i64 %v, i64* %p, align 8
  ret void
}

; The default alignment in LLVM is the same as the defualt alignment in wasm.

; CHECK-LABEL: sti64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64(i64 *%p, i64 %v) {
  store i64 %v, i64* %p
  ret void
}

; CHECK-LABEL: sti64_a16:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a16(i64 *%p, i64 %v) {
  store i64 %v, i64* %p, align 16
  ret void
}

; Truncating stores.

; CHECK-LABEL: sti8_a1:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store8 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti8_a1(i8 *%p, i64 %w) {
  %v = trunc i64 %w to i8
  store i8 %v, i8* %p, align 1
  ret void
}

; CHECK-LABEL: sti8_a2:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store8 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti8_a2(i8 *%p, i64 %w) {
  %v = trunc i64 %w to i8
  store i8 %v, i8* %p, align 2
  ret void
}

; CHECK-LABEL: sti16_a1:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store16 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a1(i16 *%p, i64 %w) {
  %v = trunc i64 %w to i16
  store i16 %v, i16* %p, align 1
  ret void
}

; CHECK-LABEL: sti16_a2:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a2(i16 *%p, i64 %w) {
  %v = trunc i64 %w to i16
  store i16 %v, i16* %p, align 2
  ret void
}

; CHECK-LABEL: sti16_a4:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a4(i16 *%p, i64 %w) {
  %v = trunc i64 %w to i16
  store i16 %v, i16* %p, align 4
  ret void
}

; CHECK-LABEL: sti32_a1:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store32 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a1(i32 *%p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, i32* %p, align 1
  ret void
}

; CHECK-LABEL: sti32_a2:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store32 0($0):p2align=1, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a2(i32 *%p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, i32* %p, align 2
  ret void
}

; CHECK-LABEL: sti32_a4:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store32 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a4(i32 *%p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, i32* %p, align 4
  ret void
}

; CHECK-LABEL: sti32_a8:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store32 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a8(i32 *%p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, i32* %p, align 8
  ret void
}
