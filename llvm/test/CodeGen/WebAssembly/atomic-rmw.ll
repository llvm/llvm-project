; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+atomics,+sign-ext | FileCheck %s

; Test atomic RMW (read-modify-write) instructions are assembled properly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Atomic read-modify-writes: 32-bit
;===----------------------------------------------------------------------------

; CHECK-LABEL: add_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: sub_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub_i32(i32* %p, i32 %v) {
  %old = atomicrmw sub i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: and_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and_i32(i32* %p, i32 %v) {
  %old = atomicrmw and i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: or_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or_i32(i32* %p, i32 %v) {
  %old = atomicrmw or i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: xor_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor_i32(i32* %p, i32 %v) {
  %old = atomicrmw xor i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: xchg_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xchg_i32(i32* %p, i32 %v) {
  %old = atomicrmw xchg i32* %p, i32 %v seq_cst
  ret i32 %old
}

;===----------------------------------------------------------------------------
; Atomic read-modify-writes: 64-bit
;===----------------------------------------------------------------------------

; CHECK-LABEL: add_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_i64(i64* %p, i64 %v) {
  %old = atomicrmw add i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: sub_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_i64(i64* %p, i64 %v) {
  %old = atomicrmw sub i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: and_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_i64(i64* %p, i64 %v) {
  %old = atomicrmw and i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: or_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_i64(i64* %p, i64 %v) {
  %old = atomicrmw or i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: xor_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_i64(i64* %p, i64 %v) {
  %old = atomicrmw xor i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: xchg_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_i64(i64* %p, i64 %v) {
  %old = atomicrmw xchg i64* %p, i64 %v seq_cst
  ret i64 %old
}

;===----------------------------------------------------------------------------
; Atomic truncating & sign-extending RMWs
;===----------------------------------------------------------------------------

; add

; CHECK-LABEL: add_sext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @add_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_sext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @add_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_sext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @add_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: add_sext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @add_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.add, i64_extend_s/i32
; CHECK-LABEL: add_sext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i32.wrap/i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.add $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @add_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw add i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; sub

; CHECK-LABEL: sub_sext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @sub_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_sext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @sub_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_sext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sub_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: sub_sext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sub_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.sub, i64_extend_s/i32
; CHECK-LABEL: sub_sext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i32.wrap/i64 $push0=, $1
; CHECK: i32.atomic.rmw.sub $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @sub_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw sub i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; and

; CHECK-LABEL: and_sext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @and_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_sext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @and_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_sext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @and_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: and_sext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @and_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.and, i64_extend_s/i32
; CHECK-LABEL: and_sext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i32.wrap/i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.and $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @and_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw and i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; or

; CHECK-LABEL: or_sext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @or_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_sext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @or_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_sext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @or_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: or_sext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @or_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.or, i64_extend_s/i32
; CHECK-LABEL: or_sext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i32.wrap/i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.or $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @or_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw or i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; xor

; CHECK-LABEL: xor_sext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xor_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_sext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xor_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_sext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xor_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xor_sext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xor_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.xor, i64_extend_s/i32
; CHECK-LABEL: xor_sext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i32.wrap/i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.xor $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @xor_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xor i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; xchg

; CHECK-LABEL: xchg_sext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xchg_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_sext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xchg_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_sext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xchg_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xchg_sext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xchg_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.xchg, i64_extend_s/i32
; CHECK-LABEL: xchg_sext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i32.wrap/i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.xchg $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @xchg_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xchg i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

;===----------------------------------------------------------------------------
; Atomic truncating & zero-extending RMWs
;===----------------------------------------------------------------------------

; add

; CHECK-LABEL: add_zext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_zext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_zext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: add_zext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: add_zext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw32_u.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw add i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; sub

; CHECK-LABEL: sub_zext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_zext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_zext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: sub_zext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: sub_zext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw32_u.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw sub i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; and

; CHECK-LABEL: and_zext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_zext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_zext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: and_zext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: and_zext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw32_u.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw and i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; or

; CHECK-LABEL: or_zext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_zext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_zext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: or_zext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: or_zext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw32_u.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw or i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; xor

; CHECK-LABEL: xor_zext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_zext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_zext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xor_zext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xor_zext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw32_u.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xor i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; xchg

; CHECK-LABEL: xchg_zext_i8_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw8_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xchg_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_zext_i16_i32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.atomic.rmw16_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xchg_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_zext_i8_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw8_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xchg_zext_i16_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw16_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xchg_zext_i32_i64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK: i64.atomic.rmw32_u.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xchg i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}
