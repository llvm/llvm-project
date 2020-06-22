; RUN: llc < %s --mtriple=wasm32-unknown-unknown -mattr=+atomics,+sign-ext -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck --check-prefixes CHECK,CHK32 %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -mattr=+atomics,+sign-ext -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck --check-prefixes CHECK,CHK64 %s

; Test that extending loads are assembled properly.

; CHECK-LABEL: sext_i8_i32:
; CHECK: i32.atomic.load8_u $push0=, 0($0){{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @sext_i8_i32(i8 *%p) {
  %v = load atomic i8, i8* %p seq_cst, align 1
  %e = sext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i8_i32:
; CHECK: i32.atomic.load8_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @zext_i8_i32(i8 *%p) {
e1:
  %v = load atomic i8, i8* %p seq_cst, align 1
  %e = zext i8 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i16_i32:
; CHECK: i32.atomic.load16_u $push0=, 0($0){{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @sext_i16_i32(i16 *%p) {
  %v = load atomic i16, i16* %p seq_cst, align 2
  %e = sext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: zext_i16_i32:
; CHECK: i32.atomic.load16_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @zext_i16_i32(i16 *%p) {
  %v = load atomic i16, i16* %p seq_cst, align 2
  %e = zext i16 %v to i32
  ret i32 %e
}

; CHECK-LABEL: sext_i8_i64:
; CHECK: i64.atomic.load8_u $push0=, 0($0){{$}}
; CHECK: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sext_i8_i64(i8 *%p) {
  %v = load atomic i8, i8* %p seq_cst, align 1
  %e = sext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i8_i64:
; CHECK: i64.atomic.load8_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @zext_i8_i64(i8 *%p) {
  %v = load atomic i8, i8* %p seq_cst, align 1
  %e = zext i8 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i16_i64:
; CHECK: i64.atomic.load16_u $push0=, 0($0){{$}}
; CHECK: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sext_i16_i64(i16 *%p) {
  %v = load atomic i16, i16* %p seq_cst, align 2
  %e = sext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i16_i64:
; CHECK: i64.atomic.load16_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @zext_i16_i64(i16 *%p) {
  %v = load atomic i16, i16* %p seq_cst, align 2
  %e = zext i16 %v to i64
  ret i64 %e
}

; CHECK-LABEL: sext_i32_i64:
; CHECK: i32.atomic.load $push0=, 0($0){{$}}
; CHECK: i64.extend_i32_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sext_i32_i64(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  %e = sext i32 %v to i64
  ret i64 %e
}

; CHECK-LABEL: zext_i32_i64:
; CHECK: i64.atomic.load32_u $push0=, 0($0){{$}}
; CHECK: return $pop0{{$}}
define i64 @zext_i32_i64(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  %e = zext i32 %v to i64
  ret i64 %e
}
