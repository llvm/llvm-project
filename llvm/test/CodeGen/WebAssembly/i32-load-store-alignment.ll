; RUN: llc < %s -mattr=+atomics -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s

; Test loads and stores with custom alignment values.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Loads
;===----------------------------------------------------------------------------

; CHECK-LABEL: ldi32_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32_a1(i32 *%p) {
  %v = load i32, i32* %p, align 1
  ret i32 %v
}

; CHECK-LABEL: ldi32_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load $push[[NUM:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32_a2(i32 *%p) {
  %v = load i32, i32* %p, align 2
  ret i32 %v
}

; 4 is the default alignment for i32 so no attribute is needed.

; CHECK-LABEL: ldi32_a4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32_a4(i32 *%p) {
  %v = load i32, i32* %p, align 4
  ret i32 %v
}

; The default alignment in LLVM is the same as the defualt alignment in wasm.

; CHECK-LABEL: ldi32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32(i32 *%p) {
  %v = load i32, i32* %p
  ret i32 %v
}

; 8 is greater than the default alignment so it is ignored.

; CHECK-LABEL: ldi32_a8:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32_a8(i32 *%p) {
  %v = load i32, i32* %p, align 8
  ret i32 %v
}

;===----------------------------------------------------------------------------
; Extending loads
;===----------------------------------------------------------------------------

; CHECK-LABEL: ldi8_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i8 @ldi8_a1(i8 *%p) {
  %v = load i8, i8* %p, align 1
  ret i8 %v
}

; CHECK-LABEL: ldi8_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i8 @ldi8_a2(i8 *%p) {
  %v = load i8, i8* %p, align 2
  ret i8 %v
}

; CHECK-LABEL: ldi16_a1:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load16_u $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i16 @ldi16_a1(i16 *%p) {
  %v = load i16, i16* %p, align 1
  ret i16 %v
}

; CHECK-LABEL: ldi16_a2:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load16_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i16 @ldi16_a2(i16 *%p) {
  %v = load i16, i16* %p, align 2
  ret i16 %v
}

; CHECK-LABEL: ldi16_a4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.load16_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i16 @ldi16_a4(i16 *%p) {
  %v = load i16, i16* %p, align 4
  ret i16 %v
}

;===----------------------------------------------------------------------------
; Stores
;===----------------------------------------------------------------------------

; CHECK-LABEL: sti32_a1:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a1(i32 *%p, i32 %v) {
  store i32 %v, i32* %p, align 1
  ret void
}

; CHECK-LABEL: sti32_a2:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store 0($0):p2align=1, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a2(i32 *%p, i32 %v) {
  store i32 %v, i32* %p, align 2
  ret void
}

; 4 is the default alignment for i32 so no attribute is needed.

; CHECK-LABEL: sti32_a4:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a4(i32 *%p, i32 %v) {
  store i32 %v, i32* %p, align 4
  ret void
}

; The default alignment in LLVM is the same as the defualt alignment in wasm.

; CHECK-LABEL: sti32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32(i32 *%p, i32 %v) {
  store i32 %v, i32* %p
  ret void
}

; CHECK-LABEL: sti32_a8:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a8(i32 *%p, i32 %v) {
  store i32 %v, i32* %p, align 8
  ret void
}

;===----------------------------------------------------------------------------
; Truncating stores
;===----------------------------------------------------------------------------

; CHECK-LABEL: sti8_a1:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store8 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti8_a1(i8 *%p, i8 %v) {
  store i8 %v, i8* %p, align 1
  ret void
}

; CHECK-LABEL: sti8_a2:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store8 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti8_a2(i8 *%p, i8 %v) {
  store i8 %v, i8* %p, align 2
  ret void
}

; CHECK-LABEL: sti16_a1:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store16 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a1(i16 *%p, i16 %v) {
  store i16 %v, i16* %p, align 1
  ret void
}

; CHECK-LABEL: sti16_a2:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a2(i16 *%p, i16 %v) {
  store i16 %v, i16* %p, align 2
  ret void
}

; CHECK-LABEL: sti16_a4:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a4(i16 *%p, i16 %v) {
  store i16 %v, i16* %p, align 4
  ret void
}

;===----------------------------------------------------------------------------
; Atomic loads
;===----------------------------------------------------------------------------

; Wasm atomics have the alignment field, but it must always have the type's
; natural alignment.

; CHECK-LABEL: ldi32_atomic_a4:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32_atomic_a4(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %v
}

; 8 is greater than the default alignment so it is ignored.

; CHECK-LABEL: ldi32_atomic_a8:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32_atomic_a8(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 8
  ret i32 %v
}

;===----------------------------------------------------------------------------
; Atomic stores
;===----------------------------------------------------------------------------

; CHECK-LABEL: sti32_atomic_a4:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_atomic_a4(i32 *%p, i32 %v) {
 store atomic i32 %v, i32* %p seq_cst, align 4
 ret void
}

; 8 is greater than the default alignment so it is ignored.

; CHECK-LABEL: sti32_atomic_a8:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_atomic_a8(i32 *%p, i32 %v) {
 store atomic i32 %v, i32* %p seq_cst, align 8
 ret void
}
