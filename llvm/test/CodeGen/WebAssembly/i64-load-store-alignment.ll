; RUN: llc < %s -mattr=+atomics -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test loads and stores with custom alignment values.

target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Loads
;===----------------------------------------------------------------------------

; CHECK-LABEL: ldi64_a1:
; CHECK-NEXT: .functype ldi64_a1 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a1(ptr %p) {
  %v = load i64, ptr %p, align 1
  ret i64 %v
}

; CHECK-LABEL: ldi64_a2:
; CHECK-NEXT: .functype ldi64_a2 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a2(ptr %p) {
  %v = load i64, ptr %p, align 2
  ret i64 %v
}

; CHECK-LABEL: ldi64_a4:
; CHECK-NEXT: .functype ldi64_a4 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a4(ptr %p) {
  %v = load i64, ptr %p, align 4
  ret i64 %v
}

; 8 is the default alignment for i64 so no attribute is needed.

; CHECK-LABEL: ldi64_a8:
; CHECK-NEXT: .functype ldi64_a8 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a8(ptr %p) {
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

; The default alignment in LLVM is the same as the default alignment in wasm.

; CHECK-LABEL: ldi64:
; CHECK-NEXT: .functype ldi64 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64(ptr %p) {
  %v = load i64, ptr %p
  ret i64 %v
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: ldi64_a16:
; CHECK-NEXT: .functype ldi64_a16 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_a16(ptr %p) {
  %v = load i64, ptr %p, align 16
  ret i64 %v
}

;===----------------------------------------------------------------------------
; Extending loads
;===----------------------------------------------------------------------------

; CHECK-LABEL: ldi8_a1:
; CHECK-NEXT: .functype ldi8_a1 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi8_a1(ptr %p) {
  %v = load i8, ptr %p, align 1
  %w = zext i8 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi8_a2:
; CHECK-NEXT: .functype ldi8_a2 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi8_a2(ptr %p) {
  %v = load i8, ptr %p, align 2
  %w = zext i8 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi16_a1:
; CHECK-NEXT: .functype ldi16_a1 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load16_u $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi16_a1(ptr %p) {
  %v = load i16, ptr %p, align 1
  %w = zext i16 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi16_a2:
; CHECK-NEXT: .functype ldi16_a2 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load16_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi16_a2(ptr %p) {
  %v = load i16, ptr %p, align 2
  %w = zext i16 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi16_a4:
; CHECK-NEXT: .functype ldi16_a4 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load16_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi16_a4(ptr %p) {
  %v = load i16, ptr %p, align 4
  %w = zext i16 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a1:
; CHECK-NEXT: .functype ldi32_a1 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a1(ptr %p) {
  %v = load i32, ptr %p, align 1
  %w = zext i32 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a2:
; CHECK-NEXT: .functype ldi32_a2 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a2(ptr %p) {
  %v = load i32, ptr %p, align 2
  %w = zext i32 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a4:
; CHECK-NEXT: .functype ldi32_a4 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a4(ptr %p) {
  %v = load i32, ptr %p, align 4
  %w = zext i32 %v to i64
  ret i64 %w
}

; CHECK-LABEL: ldi32_a8:
; CHECK-NEXT: .functype ldi32_a8 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.load32_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi32_a8(ptr %p) {
  %v = load i32, ptr %p, align 8
  %w = zext i32 %v to i64
  ret i64 %w
}

;===----------------------------------------------------------------------------
; Stores
;===----------------------------------------------------------------------------

; CHECK-LABEL: sti64_a1:
; CHECK-NEXT: .functype sti64_a1 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a1(ptr %p, i64 %v) {
  store i64 %v, ptr %p, align 1
  ret void
}

; CHECK-LABEL: sti64_a2:
; CHECK-NEXT: .functype sti64_a2 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store 0($0):p2align=1, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a2(ptr %p, i64 %v) {
  store i64 %v, ptr %p, align 2
  ret void
}

; CHECK-LABEL: sti64_a4:
; CHECK-NEXT: .functype sti64_a4 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a4(ptr %p, i64 %v) {
  store i64 %v, ptr %p, align 4
  ret void
}

; 8 is the default alignment for i32 so no attribute is needed.

; CHECK-LABEL: sti64_a8:
; CHECK-NEXT: .functype sti64_a8 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a8(ptr %p, i64 %v) {
  store i64 %v, ptr %p, align 8
  ret void
}

; The default alignment in LLVM is the same as the default alignment in wasm.

; CHECK-LABEL: sti64:
; CHECK-NEXT: .functype sti64 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64(ptr %p, i64 %v) {
  store i64 %v, ptr %p
  ret void
}

; CHECK-LABEL: sti64_a16:
; CHECK-NEXT: .functype sti64_a16 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_a16(ptr %p, i64 %v) {
  store i64 %v, ptr %p, align 16
  ret void
}

;===----------------------------------------------------------------------------
; Truncating stores
;===----------------------------------------------------------------------------

; CHECK-LABEL: sti8_a1:
; CHECK-NEXT: .functype sti8_a1 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store8 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti8_a1(ptr %p, i64 %w) {
  %v = trunc i64 %w to i8
  store i8 %v, ptr %p, align 1
  ret void
}

; CHECK-LABEL: sti8_a2:
; CHECK-NEXT: .functype sti8_a2 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store8 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti8_a2(ptr %p, i64 %w) {
  %v = trunc i64 %w to i8
  store i8 %v, ptr %p, align 2
  ret void
}

; CHECK-LABEL: sti16_a1:
; CHECK-NEXT: .functype sti16_a1 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store16 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a1(ptr %p, i64 %w) {
  %v = trunc i64 %w to i16
  store i16 %v, ptr %p, align 1
  ret void
}

; CHECK-LABEL: sti16_a2:
; CHECK-NEXT: .functype sti16_a2 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a2(ptr %p, i64 %w) {
  %v = trunc i64 %w to i16
  store i16 %v, ptr %p, align 2
  ret void
}

; CHECK-LABEL: sti16_a4:
; CHECK-NEXT: .functype sti16_a4 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti16_a4(ptr %p, i64 %w) {
  %v = trunc i64 %w to i16
  store i16 %v, ptr %p, align 4
  ret void
}

; CHECK-LABEL: sti32_a1:
; CHECK-NEXT: .functype sti32_a1 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store32 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a1(ptr %p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, ptr %p, align 1
  ret void
}

; CHECK-LABEL: sti32_a2:
; CHECK-NEXT: .functype sti32_a2 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store32 0($0):p2align=1, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a2(ptr %p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, ptr %p, align 2
  ret void
}

; CHECK-LABEL: sti32_a4:
; CHECK-NEXT: .functype sti32_a4 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store32 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a4(ptr %p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, ptr %p, align 4
  ret void
}

; CHECK-LABEL: sti32_a8:
; CHECK-NEXT: .functype sti32_a8 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.store32 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32_a8(ptr %p, i64 %w) {
  %v = trunc i64 %w to i32
  store i32 %v, ptr %p, align 8
  ret void
}

;===----------------------------------------------------------------------------
; Atomic loads
;===----------------------------------------------------------------------------

; Wasm atomics have the alignment field, but it must always have the type's
; natural alignment.

; CHECK-LABEL: ldi64_atomic_a8:
; CHECK-NEXT: .functype ldi64_atomic_a8 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_atomic_a8(ptr %p) {
  %v = load atomic i64, ptr %p seq_cst, align 8
  ret i64 %v
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: ldi64_atomic_a16:
; CHECK-NEXT: .functype ldi64_atomic_a16 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64_atomic_a16(ptr %p) {
  %v = load atomic i64, ptr %p seq_cst, align 16
  ret i64 %v
}

;===----------------------------------------------------------------------------
; Atomic stores
;===----------------------------------------------------------------------------

; CHECK-LABEL: sti64_atomic_a4:
; CHECK-NEXT: .functype sti64_atomic_a4 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_atomic_a4(ptr %p, i64 %v) {
 store atomic i64 %v, ptr %p seq_cst, align 8
 ret void
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: sti64_atomic_a8:
; CHECK-NEXT: .functype sti64_atomic_a8 (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64_atomic_a8(ptr %p, i64 %v) {
 store atomic i64 %v, ptr %p seq_cst, align 16
 ret void
}
