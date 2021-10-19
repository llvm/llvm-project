; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics,+sign-ext | FileCheck %s

; Test atomic RMW (read-modify-write) instructions are assembled properly.

target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Atomic read-modify-writes: 32-bit
;===----------------------------------------------------------------------------

; CHECK-LABEL: add_i32:
; CHECK-NEXT: .functype add_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_i32(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: sub_i32:
; CHECK-NEXT: .functype sub_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub_i32(i32* %p, i32 %v) {
  %old = atomicrmw sub i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: and_i32:
; CHECK-NEXT: .functype and_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and_i32(i32* %p, i32 %v) {
  %old = atomicrmw and i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: or_i32:
; CHECK-NEXT: .functype or_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or_i32(i32* %p, i32 %v) {
  %old = atomicrmw or i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: xor_i32:
; CHECK-NEXT: .functype xor_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor_i32(i32* %p, i32 %v) {
  %old = atomicrmw xor i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: xchg_i32:
; CHECK-NEXT: .functype xchg_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xchg_i32(i32* %p, i32 %v) {
  %old = atomicrmw xchg i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_loaded_value:
; CHECK-NEXT: .functype cmpxchg_i32_loaded_value (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_loaded_value(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_success:
; CHECK-NEXT: .functype cmpxchg_i32_success (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: i32.eq $push1=, $pop0, $1{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i1 @cmpxchg_i32_success(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new seq_cst seq_cst
  %succ = extractvalue { i32, i1 } %pair, 1
  ret i1 %succ
}

; Unsupported instructions are expanded using cmpxchg with a loop.

; CHECK-LABEL: nand_i32:
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i32 @nand_i32(i32* %p, i32 %v) {
  %old = atomicrmw nand i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: max_i32:
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i32 @max_i32(i32* %p, i32 %v) {
  %old = atomicrmw max i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: min_i32:
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i32 @min_i32(i32* %p, i32 %v) {
  %old = atomicrmw min i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: umax_i32:
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i32 @umax_i32(i32* %p, i32 %v) {
  %old = atomicrmw umax i32* %p, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: umin_i32:
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i32 @umin_i32(i32* %p, i32 %v) {
  %old = atomicrmw umin i32* %p, i32 %v seq_cst
  ret i32 %old
}

;===----------------------------------------------------------------------------
; Atomic read-modify-writes: 64-bit
;===----------------------------------------------------------------------------

; CHECK-LABEL: add_i64:
; CHECK-NEXT: .functype add_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_i64(i64* %p, i64 %v) {
  %old = atomicrmw add i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: sub_i64:
; CHECK-NEXT: .functype sub_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.sub $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_i64(i64* %p, i64 %v) {
  %old = atomicrmw sub i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: and_i64:
; CHECK-NEXT: .functype and_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.and $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_i64(i64* %p, i64 %v) {
  %old = atomicrmw and i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: or_i64:
; CHECK-NEXT: .functype or_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.or $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_i64(i64* %p, i64 %v) {
  %old = atomicrmw or i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: xor_i64:
; CHECK-NEXT: .functype xor_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.xor $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_i64(i64* %p, i64 %v) {
  %old = atomicrmw xor i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: xchg_i64:
; CHECK-NEXT: .functype xchg_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.xchg $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_i64(i64* %p, i64 %v) {
  %old = atomicrmw xchg i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: cmpxchg_i64_loaded_value:
; CHECK-NEXT: .functype cmpxchg_i64_loaded_value (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @cmpxchg_i64_loaded_value(i64* %p, i64 %exp, i64 %new) {
  %pair = cmpxchg i64* %p, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

; CHECK-LABEL: cmpxchg_i64_success:
; CHECK-NEXT: .functype cmpxchg_i64_success (i32, i64, i64) -> (i32){{$}}
; CHECK: i64.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: i64.eq $push1=, $pop0, $1{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i1 @cmpxchg_i64_success(i64* %p, i64 %exp, i64 %new) {
  %pair = cmpxchg i64* %p, i64 %exp, i64 %new seq_cst seq_cst
  %succ = extractvalue { i64, i1 } %pair, 1
  ret i1 %succ
}

; Unsupported instructions are expanded using cmpxchg with a loop.

; CHECK-LABEL: nand_i64:
; CHECK: loop
; CHECK: i64.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i64 @nand_i64(i64* %p, i64 %v) {
  %old = atomicrmw nand i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: max_i64:
; CHECK: loop
; CHECK: i64.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i64 @max_i64(i64* %p, i64 %v) {
  %old = atomicrmw max i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: min_i64:
; CHECK: loop
; CHECK: i64.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i64 @min_i64(i64* %p, i64 %v) {
  %old = atomicrmw min i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: umax_i64:
; CHECK: loop
; CHECK: i64.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i64 @umax_i64(i64* %p, i64 %v) {
  %old = atomicrmw umax i64* %p, i64 %v seq_cst
  ret i64 %old
}

; CHECK-LABEL: umin_i64:
; CHECK: loop
; CHECK: i64.atomic.rmw.cmpxchg
; CHECK: br_if 0
; CHECK: end_loop
define i64 @umin_i64(i64* %p, i64 %v) {
  %old = atomicrmw umin i64* %p, i64 %v seq_cst
  ret i64 %old
}

;===----------------------------------------------------------------------------
; Atomic truncating & sign-extending RMWs
;===----------------------------------------------------------------------------

; add

; CHECK-LABEL: add_sext_i8_i32:
; CHECK-NEXT: .functype add_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @add_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_sext_i16_i32:
; CHECK-NEXT: .functype add_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @add_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_sext_i8_i64:
; CHECK-NEXT: .functype add_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @add_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: add_sext_i16_i64:
; CHECK-NEXT: .functype add_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @add_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.add, i64.extend_i32_s
; CHECK-LABEL: add_sext_i32_i64:
; CHECK-NEXT: .functype add_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.add $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @add_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw add i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; sub

; CHECK-LABEL: sub_sext_i8_i32:
; CHECK-NEXT: .functype sub_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @sub_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_sext_i16_i32:
; CHECK-NEXT: .functype sub_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @sub_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_sext_i8_i64:
; CHECK-NEXT: .functype sub_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sub_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: sub_sext_i16_i64:
; CHECK-NEXT: .functype sub_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @sub_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.sub, i64.extend_i32_s
; CHECK-LABEL: sub_sext_i32_i64:
; CHECK-NEXT: .functype sub_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push0=, $1
; CHECK: i32.atomic.rmw.sub $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @sub_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw sub i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; and

; CHECK-LABEL: and_sext_i8_i32:
; CHECK-NEXT: .functype and_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @and_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_sext_i16_i32:
; CHECK-NEXT: .functype and_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @and_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_sext_i8_i64:
; CHECK-NEXT: .functype and_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @and_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: and_sext_i16_i64:
; CHECK-NEXT: .functype and_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @and_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.and, i64.extend_i32_s
; CHECK-LABEL: and_sext_i32_i64:
; CHECK-NEXT: .functype and_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.and $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @and_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw and i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; or

; CHECK-LABEL: or_sext_i8_i32:
; CHECK-NEXT: .functype or_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @or_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_sext_i16_i32:
; CHECK-NEXT: .functype or_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @or_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_sext_i8_i64:
; CHECK-NEXT: .functype or_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @or_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: or_sext_i16_i64:
; CHECK-NEXT: .functype or_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @or_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.or, i64.extend_i32_s
; CHECK-LABEL: or_sext_i32_i64:
; CHECK-NEXT: .functype or_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.or $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @or_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw or i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; xor

; CHECK-LABEL: xor_sext_i8_i32:
; CHECK-NEXT: .functype xor_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xor_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_sext_i16_i32:
; CHECK-NEXT: .functype xor_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xor_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_sext_i8_i64:
; CHECK-NEXT: .functype xor_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xor_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xor_sext_i16_i64:
; CHECK-NEXT: .functype xor_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xor_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.xor, i64.extend_i32_s
; CHECK-LABEL: xor_sext_i32_i64:
; CHECK-NEXT: .functype xor_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.xor $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @xor_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xor i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; xchg

; CHECK-LABEL: xchg_sext_i8_i32:
; CHECK-NEXT: .functype xchg_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xchg_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_sext_i16_i32:
; CHECK-NEXT: .functype xchg_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @xchg_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_sext_i8_i64:
; CHECK-NEXT: .functype xchg_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xchg_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xchg_sext_i16_i64:
; CHECK-NEXT: .functype xchg_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @xchg_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.xchg, i64.extend_i32_s
; CHECK-LABEL: xchg_sext_i32_i64:
; CHECK-NEXT: .functype xchg_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push0=, $1{{$}}
; CHECK: i32.atomic.rmw.xchg $push1=, 0($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
; CHECK-NEXT: return $pop2{{$}}
define i64 @xchg_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xchg i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

; cmpxchg

; CHECK-LABEL: cmpxchg_sext_i8_i32:
; CHECK-NEXT: .functype cmpxchg_sext_i8_i32 (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @cmpxchg_sext_i8_i32(i8* %p, i32 %exp, i32 %new) {
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %p, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: cmpxchg_sext_i16_i32:
; CHECK-NEXT: .functype cmpxchg_sext_i16_i32 (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i32 @cmpxchg_sext_i16_i32(i16* %p, i32 %exp, i32 %new) {
  %exp_t = trunc i32 %exp to i16
  %new_t = trunc i32 %new to i16
  %pair = cmpxchg i16* %p, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %e = sext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: cmpxchg_sext_i8_i64:
; CHECK-NEXT: .functype cmpxchg_sext_i8_i64 (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: i64.extend8_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @cmpxchg_sext_i8_i64(i8* %p, i64 %exp, i64 %new) {
  %exp_t = trunc i64 %exp to i8
  %new_t = trunc i64 %new to i8
  %pair = cmpxchg i8* %p, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %e = sext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: cmpxchg_sext_i16_i64:
; CHECK-NEXT: .functype cmpxchg_sext_i16_i64 (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define i64 @cmpxchg_sext_i16_i64(i16* %p, i64 %exp, i64 %new) {
  %exp_t = trunc i64 %exp to i16
  %new_t = trunc i64 %new to i16
  %pair = cmpxchg i16* %p, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.cmpxchg, i64.extend_i32_s
; CHECK-LABEL: cmpxchg_sext_i32_i64:
; CHECK-NEXT: .functype cmpxchg_sext_i32_i64 (i32, i64, i64) -> (i64){{$}}
; CHECK: i32.wrap_i64 $push1=, $1{{$}}
; CHECK-NEXT: i32.wrap_i64 $push0=, $2{{$}}
; CHECK-NEXT: i32.atomic.rmw.cmpxchg $push2=, 0($0), $pop1, $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push3=, $pop2{{$}}
; CHECK-NEXT: return $pop3{{$}}
define i64 @cmpxchg_sext_i32_i64(i32* %p, i64 %exp, i64 %new) {
  %exp_t = trunc i64 %exp to i32
  %new_t = trunc i64 %new to i32
  %pair = cmpxchg i32* %p, i32 %exp_t, i32 %new_t seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  %e = sext i32 %old to i64
  ret i64 %e
}

; Unsupported instructions are expanded using cmpxchg with a loop.
; Here we take a nand as an example.

; nand

; CHECK-LABEL: nand_sext_i8_i32:
; CHECK-NEXT: .functype nand_sext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw8.cmpxchg_u
; CHECK: i32.extend8_s
define i32 @nand_sext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw nand i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: nand_sext_i16_i32:
; CHECK-NEXT: .functype nand_sext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw16.cmpxchg_u
; CHECK: i32.extend16_s
define i32 @nand_sext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw nand i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i32
  ret i32 %e
}

; FIXME Currently this cannot make use of i64.atomic.rmw8.cmpxchg_u
; CHECK-LABEL: nand_sext_i8_i64:
; CHECK-NEXT: .functype nand_sext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw8.cmpxchg_u
; CHECK: i64.extend_i32_u
; CHECK: i64.extend8_s
define i64 @nand_sext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw nand i8* %p, i8 %t seq_cst
  %e = sext i8 %old to i64
  ret i64 %e
}

; FIXME Currently this cannot make use of i64.atomic.rmw16.cmpxchg_u
; CHECK-LABEL: nand_sext_i16_i64:
; CHECK-NEXT: .functype nand_sext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw16.cmpxchg_u
; CHECK: i64.extend_i32_u
; CHECK: i64.extend16_s
define i64 @nand_sext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw nand i16* %p, i16 %t seq_cst
  %e = sext i16 %old to i64
  ret i64 %e
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.nand, i64.extend_i32_s
; CHECK-LABEL: nand_sext_i32_i64:
; CHECK-NEXT: .functype nand_sext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: i64.extend_i32_s
define i64 @nand_sext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw nand i32* %p, i32 %t seq_cst
  %e = sext i32 %old to i64
  ret i64 %e
}

;===----------------------------------------------------------------------------
; Atomic truncating & zero-extending RMWs
;===----------------------------------------------------------------------------

; add

; CHECK-LABEL: add_zext_i8_i32:
; CHECK-NEXT: .functype add_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_zext_i16_i32:
; CHECK-NEXT: .functype add_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @add_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: add_zext_i8_i64:
; CHECK-NEXT: .functype add_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: add_zext_i16_i64:
; CHECK-NEXT: .functype add_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw add i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: add_zext_i32_i64:
; CHECK-NEXT: .functype add_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @add_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw add i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; sub

; CHECK-LABEL: sub_zext_i8_i32:
; CHECK-NEXT: .functype sub_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_zext_i16_i32:
; CHECK-NEXT: .functype sub_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @sub_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: sub_zext_i8_i64:
; CHECK-NEXT: .functype sub_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw sub i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: sub_zext_i16_i64:
; CHECK-NEXT: .functype sub_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw sub i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: sub_zext_i32_i64:
; CHECK-NEXT: .functype sub_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.sub_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @sub_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw sub i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; and

; CHECK-LABEL: and_zext_i8_i32:
; CHECK-NEXT: .functype and_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_zext_i16_i32:
; CHECK-NEXT: .functype and_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @and_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: and_zext_i8_i64:
; CHECK-NEXT: .functype and_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw and i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: and_zext_i16_i64:
; CHECK-NEXT: .functype and_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw and i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: and_zext_i32_i64:
; CHECK-NEXT: .functype and_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.and_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @and_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw and i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; or

; CHECK-LABEL: or_zext_i8_i32:
; CHECK-NEXT: .functype or_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_zext_i16_i32:
; CHECK-NEXT: .functype or_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @or_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: or_zext_i8_i64:
; CHECK-NEXT: .functype or_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw or i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: or_zext_i16_i64:
; CHECK-NEXT: .functype or_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw or i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: or_zext_i32_i64:
; CHECK-NEXT: .functype or_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.or_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @or_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw or i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; xor

; CHECK-LABEL: xor_zext_i8_i32:
; CHECK-NEXT: .functype xor_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_zext_i16_i32:
; CHECK-NEXT: .functype xor_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xor_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xor_zext_i8_i64:
; CHECK-NEXT: .functype xor_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xor i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xor_zext_i16_i64:
; CHECK-NEXT: .functype xor_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xor i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xor_zext_i32_i64:
; CHECK-NEXT: .functype xor_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.xor_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xor_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xor i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; xchg

; CHECK-LABEL: xchg_zext_i8_i32:
; CHECK-NEXT: .functype xchg_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xchg_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_zext_i16_i32:
; CHECK-NEXT: .functype xchg_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @xchg_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: xchg_zext_i8_i64:
; CHECK-NEXT: .functype xchg_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw xchg i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xchg_zext_i16_i64:
; CHECK-NEXT: .functype xchg_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw xchg i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: xchg_zext_i32_i64:
; CHECK-NEXT: .functype xchg_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.xchg_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @xchg_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw xchg i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}

; cmpxchg

; CHECK-LABEL: cmpxchg_zext_i8_i32:
; CHECK-NEXT: .functype cmpxchg_zext_i8_i32 (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw8.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_zext_i8_i32(i8* %p, i32 %exp, i32 %new) {
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %p, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: cmpxchg_zext_i16_i32:
; CHECK-NEXT: .functype cmpxchg_zext_i16_i32 (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw16.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_zext_i16_i32(i16* %p, i32 %exp, i32 %new) {
  %exp_t = trunc i32 %exp to i16
  %new_t = trunc i32 %new to i16
  %pair = cmpxchg i16* %p, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %e = zext i16 %old to i32
  ret i32 %e
}

; CHECK-LABEL: cmpxchg_zext_i8_i64:
; CHECK-NEXT: .functype cmpxchg_zext_i8_i64 (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw8.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @cmpxchg_zext_i8_i64(i8* %p, i64 %exp, i64 %new) {
  %exp_t = trunc i64 %exp to i8
  %new_t = trunc i64 %new to i8
  %pair = cmpxchg i8* %p, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %e = zext i8 %old to i64
  ret i64 %e
}

; CHECK-LABEL: cmpxchg_zext_i16_i64:
; CHECK-NEXT: .functype cmpxchg_zext_i16_i64 (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw16.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @cmpxchg_zext_i16_i64(i16* %p, i64 %exp, i64 %new) {
  %exp_t = trunc i64 %exp to i16
  %new_t = trunc i64 %new to i16
  %pair = cmpxchg i16* %p, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %e = zext i16 %old to i64
  ret i64 %e
}

; CHECK-LABEL: cmpxchg_zext_i32_i64:
; CHECK-NEXT: .functype cmpxchg_zext_i32_i64 (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw32.cmpxchg_u $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @cmpxchg_zext_i32_i64(i32* %p, i64 %exp, i64 %new) {
  %exp_t = trunc i64 %exp to i32
  %new_t = trunc i64 %new to i32
  %pair = cmpxchg i32* %p, i32 %exp_t, i32 %new_t seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  %e = zext i32 %old to i64
  ret i64 %e
}

; Unsupported instructions are expanded using cmpxchg with a loop.
; Here we take a nand as an example.

; nand

; CHECK-LABEL: nand_zext_i8_i32:
; CHECK-NEXT: .functype nand_zext_i8_i32 (i32, i32) -> (i32){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw8.cmpxchg_u
define i32 @nand_zext_i8_i32(i8* %p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw nand i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i32
  ret i32 %e
}

; CHECK-LABEL: nand_zext_i16_i32:
; CHECK-NEXT: .functype nand_zext_i16_i32 (i32, i32) -> (i32){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw16.cmpxchg_u
define i32 @nand_zext_i16_i32(i16* %p, i32 %v) {
  %t = trunc i32 %v to i16
  %old = atomicrmw nand i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i32
  ret i32 %e
}

; FIXME Currently this cannot make use of i64.atomic.rmw8.cmpxchg_u
; CHECK-LABEL: nand_zext_i8_i64:
; CHECK-NEXT: .functype nand_zext_i8_i64 (i32, i64) -> (i64){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw8.cmpxchg_u
; CHECK: i64.extend_i32_u
define i64 @nand_zext_i8_i64(i8* %p, i64 %v) {
  %t = trunc i64 %v to i8
  %old = atomicrmw nand i8* %p, i8 %t seq_cst
  %e = zext i8 %old to i64
  ret i64 %e
}

; FIXME Currently this cannot make use of i64.atomic.rmw16.cmpxchg_u
; CHECK-LABEL: nand_zext_i16_i64:
; CHECK-NEXT: .functype nand_zext_i16_i64 (i32, i64) -> (i64){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw16.cmpxchg_u
; CHECK: i64.extend_i32_u
define i64 @nand_zext_i16_i64(i16* %p, i64 %v) {
  %t = trunc i64 %v to i16
  %old = atomicrmw nand i16* %p, i16 %t seq_cst
  %e = zext i16 %old to i64
  ret i64 %e
}

; FIXME Currently this cannot make use of i64.atomic.rmw32.cmpxchg_u
; CHECK-LABEL: nand_zext_i32_i64:
; CHECK-NEXT: .functype nand_zext_i32_i64 (i32, i64) -> (i64){{$}}
; CHECK: loop
; CHECK: i32.atomic.rmw.cmpxchg
; CHECK: i64.extend_i32_u
define i64 @nand_zext_i32_i64(i32* %p, i64 %v) {
  %t = trunc i64 %v to i32
  %old = atomicrmw nand i32* %p, i32 %t seq_cst
  %e = zext i32 %old to i64
  ret i64 %e
}
