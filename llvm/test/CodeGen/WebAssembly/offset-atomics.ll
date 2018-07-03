; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+atomics,+sign-ext | FileCheck %s

; Test that atomic loads are assembled properly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Basic load.

; CHECK-LABEL: load_i32_no_offset:
; CHECK: i32.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @load_i32_no_offset(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_offset:
; CHECK: i32.atomic.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_gep_offset:
; CHECK: i32.atomic.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: load_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; CHECK-LABEL: load_i64_no_offset:
; CHECK: i64.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @load_i64_no_offset(i64 *%p) {
  %v = load atomic i64, i64* %p seq_cst, align 8
  ret i64 %v
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_folded_offset:
; CHECK: i64.atomic.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_folded_gep_offset:
; CHECK: i64.atomic.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; CHECK-LABEL: load_i32_with_folded_or_offset:
; CHECK: i32.atomic.load8_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}){{$}}
; CHECK-NEXT: i32.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i32 @load_i32_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load atomic i8, i8* %arrayidx seq_cst, align 8
  %conv = sext i8 %t1 to i32
  ret i32 %conv
}

; Same as above but with store.

; CHECK-LABEL: store_i32_no_offset:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_no_offset(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p seq_cst, align 4
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_folded_offset:
; CHECK: i32.atomic.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_folded_gep_offset:
; CHECK: i32.atomic.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const        $push0=, -24{{$}}
; CHECK: i32.add          $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_unfolded_offset:
; CHECK: i32.const        $push0=, 24{{$}}
; CHECK: i32.add          $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_unfolded_gep_offset:
; CHECK: i32.const        $push0=, 24{{$}}
; CHECK: i32.add          $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_no_offset:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i64_no_offset(i64 *%p, i64 %v) {
  store atomic i64 %v, i64* %p seq_cst, align 8
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_folded_offset:
; CHECK: i64.atomic.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_folded_gep_offset:
; CHECK: i64.atomic.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const        $push0=, -24{{$}}
; CHECK: i32.add          $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_unfolded_offset:
; CHECK: i32.const        $push0=, 24{{$}}
; CHECK: i32.add          $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_unfolded_gep_offset:
; CHECK: i32.const        $push0=, 24{{$}}
; CHECK: i32.add          $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load  $push1=, 42($pop0){{$}}
define i32 @load_i32_from_numeric_address() {
  %s = inttoptr i32 42 to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}


; CHECK-LABEL: load_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load  $push1=, gv($pop0){{$}}
@gv = global i32 0
define i32 @load_i32_from_global_address() {
  %t = load atomic i32, i32* @gv seq_cst, align 4
  ret i32 %t
}

; CHECK-LABEL: store_i32_to_numeric_address:
; CHECK-NEXT: i32.const        $push0=, 0{{$}}
; CHECK-NEXT: i32.const        $push1=, 0{{$}}
; CHECK-NEXT: i32.atomic.store 42($pop0), $pop1{{$}}
define void @store_i32_to_numeric_address() {
  %s = inttoptr i32 42 to i32*
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; CHECK-LABEL: store_i32_to_global_address:
; CHECK: i32.const        $push0=, 0{{$}}
; CHECK: i32.const        $push1=, 0{{$}}
; CHECK: i32.atomic.store gv($pop0), $pop1{{$}}
define void @store_i32_to_global_address() {
  store atomic i32 0, i32* @gv seq_cst, align 4
  ret void
}

; Fold an offset into a sign-extending load.

; CHECK-LABEL: load_i8_s_with_folded_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @load_i8_s_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = sext i8 %t to i32
  ret i32 %u
}

; Fold a gep offset into a sign-extending load.

; CHECK-LABEL: load_i8_s_with_folded_gep_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @load_i8_s_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = sext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_s_i64_with_folded_gep_offset:
; CHECK: i64.atomic.load16_u  $push0=, 6($0){{$}}
define i64 @load_i16_s_i64_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 3
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = zext i16 %t to i64
  ret i64 %u
}

; CHECK-LABEL: load_i64_with_folded_or_offset:
; CHECK: i64.atomic.load8_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}){{$}}
; CHECK-NEXT: i64.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i64 @load_i64_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load atomic i8, i8* %arrayidx seq_cst, align 8
  %conv = sext i8 %t1 to i64
  ret i64 %conv
}


; Fold an offset into a zero-extending load.

; CHECK-LABEL: load_i16_u_with_folded_offset:
; CHECK: i32.atomic.load16_u $push0=, 24($0){{$}}
define i32 @load_i16_u_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i16*
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = zext i16 %t to i32
  ret i32 %u
}

; Fold a gep offset into a zero-extending load.

; CHECK-LABEL: load_i8_u_with_folded_gep_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_u_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = zext i8 %t to i32
  ret i32 %u
}


; When loading from a fixed address, materialize a zero.
; As above but with extending load.

; CHECK-LABEL: load_zext_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load16_u  $push1=, 42($pop0){{$}}
define i32 @load_zext_i32_from_numeric_address() {
  %s = inttoptr i32 42 to i16*
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_sext_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load8_u  $push1=, gv8($pop0){{$}}
; CHECK-NEXT: i32.extend8_s $push2=, $pop1{{$}}
@gv8 = global i8 0
define i32 @load_sext_i32_from_global_address() {
  %t = load atomic i8, i8* @gv8 seq_cst, align 1
  %u = sext i8 %t to i32
  ret i32 %u
}

; Fold an offset into a sign-extending load.
; As above but 32 extended to 64 bit.
; CHECK-LABEL: load_i32_i64_s_with_folded_offset:
; CHECK: i32.atomic.load $push0=, 24($0){{$}}
; CHECK-NEXT: i64.extend_s/i32 $push1=, $pop0{{$}}
define i64 @load_i32_i64_s_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  %u = sext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a zero-extending load.
; As above but 32 extended to 64 bit.
; CHECK-LABEL: load_i32_i64_u_with_folded_gep_offset:
; CHECK: i64.atomic.load32_u $push0=, 96($0){{$}}
define i64 @load_i32_i64_u_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 24
  %t = load atomic i32, i32* %s seq_cst, align 4
  %u = zext i32 %t to i64
  ret i64 %u
}

; i8 return value should test anyext loads
; CHECK-LABEL: ldi8_a1:
; CHECK: i32.atomic.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i8 @ldi8_a1(i8 *%p) {
  %v = load atomic i8, i8* %p seq_cst, align 1
  ret i8 %v
}

; Fold an offset into a truncating store.

; CHECK-LABEL: store_i8_with_folded_offset:
; CHECK: i32.atomic.store8 24($0), $pop0{{$}}
define void @store_i8_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  store atomic i8 0, i8* %s seq_cst, align 1
  ret void
}

; CHECK-LABEL: store_i16_with_folded_offset:
; CHECK: i32.atomic.store16 24($0), $pop0{{$}}
define void @store_i16_with_folded_offset(i16* %p) {
  %q = ptrtoint i16* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i16*
  store atomic i16 0, i16* %s seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i8_i64_with_folded_offset:
; CHECK: i64.atomic.store8 24($0), $1{{$}}
define void @store_i8_i64_with_folded_offset(i8* %p, i64 %v) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = trunc i64 %v to i8
  store atomic i8 %t, i8* %s seq_cst, align 1
  ret void
}

; CHECK-LABEL: store_i16_i64_with_folded_offset:
; CHECK: i64.atomic.store16 24($0), $1{{$}}
define void @store_i16_i64_with_folded_offset(i16* %p, i64 %v) {
  %q = ptrtoint i16* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i16*
  %t = trunc i64 %v to i16
  store atomic i16 %t, i16* %s seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i32_i64_with_folded_offset:
; CHECK: i64.atomic.store32 24($0), $1{{$}}
define void @store_i32_i64_with_folded_offset(i32* %p, i64 %v) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = trunc i64 %v to i32
  store atomic i32 %t, i32* %s seq_cst, align 4
  ret void
}

; Fold a gep offset into a truncating store.

; CHECK-LABEL: store_i8_with_folded_gep_offset:
; CHECK: i32.atomic.store8 24($0), $pop0{{$}}
define void @store_i8_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  store atomic i8 0, i8* %s seq_cst, align 1
  ret void
}

; CHECK-LABEL: store_i16_with_folded_gep_offset:
; CHECK: i32.atomic.store16 48($0), $pop0{{$}}
define void @store_i16_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  store atomic i16 0, i16* %s seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i8_i64_with_folded_gep_offset:
; CHECK: i64.atomic.store8 24($0), $1{{$}}
define void @store_i8_i64_with_folded_gep_offset(i8* %p, i64 %v) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = trunc i64 %v to i8
  store atomic i8 %t, i8* %s seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i16_i64_with_folded_gep_offset:
; CHECK: i64.atomic.store16 48($0), $1{{$}}
define void @store_i16_i64_with_folded_gep_offset(i16* %p, i64 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i64 %v to i16
  store atomic i16 %t, i16* %s seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i32_i64_with_folded_gep_offset:
; CHECK: i64.atomic.store32 96($0), $1{{$}}
define void @store_i32_i64_with_folded_gep_offset(i32* %p, i64 %v) {
  %s = getelementptr inbounds i32, i32* %p, i32 24
  %t = trunc i64 %v to i32
  store atomic i32 %t, i32* %s seq_cst, align 4
  ret void
}

; Fold an or_is_add pattern based offset into a truncating store.

; CHECK-LABEL: store_i8_with_folded_or_offset:
; CHECK: i32.atomic.store8 2($pop{{[0-9]+}}), $pop{{[0-9]+}}{{$}}
define void @store_i8_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 2
  store atomic i8 0, i8* %arrayidx seq_cst, align 1
  ret void
}

; CHECK-LABEL: store_i16_with_folded_or_offset:
; CHECK: i32.atomic.store16 4($pop{{[0-9]+}}), $pop{{[0-9]+}}{{$}}
define void @store_i16_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i16*
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 2
  store atomic i16 0, i16* %arrayidx seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i8_i64_with_folded_or_offset:
; CHECK: i64.atomic.store8 2($pop{{[0-9]+}}), $1{{$}}
define void @store_i8_i64_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 2
  %t = trunc i64 %v to i8
  store atomic i8 %t, i8* %arrayidx seq_cst, align 1
  ret void
}

; CHECK-LABEL: store_i16_i64_with_folded_or_offset:
; CHECK: i64.atomic.store16 4($pop{{[0-9]+}}), $1{{$}}
define void @store_i16_i64_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i16*
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 2
  %t = trunc i64 %v to i16
  store atomic i16 %t, i16* %arrayidx seq_cst, align 2
  ret void
}

; CHECK-LABEL: store_i32_i64_with_folded_or_offset:
; CHECK: i64.atomic.store32 8($pop{{[0-9]+}}), $1{{$}}
define void @store_i32_i64_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i32*
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 2
  %t = trunc i64 %v to i32
  store atomic i32 %t, i32* %arrayidx seq_cst, align 4
  ret void
}
