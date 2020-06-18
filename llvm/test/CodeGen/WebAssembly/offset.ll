; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers -disable-wasm-fallthrough-return-opt | FileCheck %s

; Test constant load and store address offsets.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Loads: 32-bit
;===----------------------------------------------------------------------------

; Basic load.

; CHECK-LABEL: load_i32_no_offset:
; CHECK: i32.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_no_offset(i32 *%p) {
  %v = load i32, i32* %p
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_offset:
; CHECK: i32.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load i32, i32* %s
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_gep_offset:
; CHECK: i32.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %t = load i32, i32* %s
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %t = load i32, i32* %s
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load i32, i32* %s
  ret i32 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: load_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  %t = load i32, i32* %s
  ret i32 %t
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load  $push1=, 42($pop0){{$}}
define i32 @load_i32_from_numeric_address() {
  %s = inttoptr i32 42 to i32*
  %t = load i32, i32* %s
  ret i32 %t
}

; CHECK-LABEL: load_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load  $push1=, gv($pop0){{$}}
@gv = global i32 0
define i32 @load_i32_from_global_address() {
  %t = load i32, i32* @gv
  ret i32 %t
}

;===----------------------------------------------------------------------------
; Loads: 64-bit
;===----------------------------------------------------------------------------

; Basic load.

; CHECK-LABEL: load_i64_no_offset:
; CHECK: i64.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @load_i64_no_offset(i64 *%p) {
  %v = load i64, i64* %p
  ret i64 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i64_with_folded_offset:
; CHECK: i64.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load i64, i64* %s
  ret i64 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i64_with_folded_gep_offset:
; CHECK: i64.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %t = load i64, i64* %s
  ret i64 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %t = load i64, i64* %s
  ret i64 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load i64, i64* %s
  ret i64 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: load_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  %t = load i64, i64* %s
  ret i64 %t
}

;===----------------------------------------------------------------------------
; Stores: 32-bit
;===----------------------------------------------------------------------------

; Basic store.

; CHECK-LABEL: store_i32_no_offset:
; CHECK-NEXT: .functype store_i32_no_offset (i32, i32) -> (){{$}}
; CHECK-NEXT: i32.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_no_offset(i32 *%p, i32 %v) {
  store i32 %v, i32* %p
  ret void
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: store_i32_with_folded_offset:
; CHECK: i32.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store i32 0, i32* %s
  ret void
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: store_i32_with_folded_gep_offset:
; CHECK: i32.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  store i32 0, i32* %s
  ret void
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: store_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  store i32 0, i32* %s
  ret void
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: store_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store i32 0, i32* %s
  ret void
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: store_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  store i32 0, i32* %s
  ret void
}

; When storing from a fixed address, materialize a zero.

; CHECK-LABEL: store_i32_to_numeric_address:
; CHECK:      i32.const $push0=, 0{{$}}
; CHECK-NEXT: i32.const $push1=, 0{{$}}
; CHECK-NEXT: i32.store 42($pop0), $pop1{{$}}
define void @store_i32_to_numeric_address() {
  %s = inttoptr i32 42 to i32*
  store i32 0, i32* %s
  ret void
}

; CHECK-LABEL: store_i32_to_global_address:
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.const $push1=, 0{{$}}
; CHECK: i32.store gv($pop0), $pop1{{$}}
define void @store_i32_to_global_address() {
  store i32 0, i32* @gv
  ret void
}

;===----------------------------------------------------------------------------
; Stores: 64-bit
;===----------------------------------------------------------------------------

; Basic store.

; CHECK-LABEL: store_i64_with_folded_offset:
; CHECK: i64.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store i64 0, i64* %s
  ret void
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: store_i64_with_folded_gep_offset:
; CHECK: i64.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  store i64 0, i64* %s
  ret void
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: store_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  store i64 0, i64* %s
  ret void
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: store_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store i64 0, i64* %s
  ret void
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: store_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  store i64 0, i64* %s
  ret void
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: store_i32_with_folded_or_offset:
; CHECK: i32.store8 2($pop{{[0-9]+}}), $pop{{[0-9]+}}{{$}}
define void @store_i32_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  store i8 0, i8* %arrayidx, align 1
  ret void
}

;===----------------------------------------------------------------------------
; Sign-extending loads
;===----------------------------------------------------------------------------

; Fold an offset into a sign-extending load.

; CHECK-LABEL: load_i8_i32_s_with_folded_offset:
; CHECK: i32.load8_s $push0=, 24($0){{$}}
define i32 @load_i8_i32_s_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load i8, i8* %s
  %u = sext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i32_i64_s_with_folded_offset:
; CHECK: i64.load32_s $push0=, 24($0){{$}}
define i64 @load_i32_i64_s_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load i32, i32* %s
  %u = sext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a sign-extending load.

; CHECK-LABEL: load_i8_i32_s_with_folded_gep_offset:
; CHECK: i32.load8_s $push0=, 24($0){{$}}
define i32 @load_i8_i32_s_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load i8, i8* %s
  %u = sext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i32_s_with_folded_gep_offset:
; CHECK: i32.load16_s $push0=, 48($0){{$}}
define i32 @load_i16_i32_s_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = load i16, i16* %s
  %u = sext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i64_s_with_folded_gep_offset:
; CHECK: i64.load16_s $push0=, 48($0){{$}}
define i64 @load_i16_i64_s_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = load i16, i16* %s
  %u = sext i16 %t to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: load_i8_i32_s_with_folded_or_offset:
; CHECK: i32.load8_s $push{{[0-9]+}}=, 2($pop{{[0-9]+}}){{$}}
define i32 @load_i8_i32_s_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load i8, i8* %arrayidx
  %conv = sext i8 %t1 to i32
  ret i32 %conv
}

; CHECK-LABEL: load_i8_i64_s_with_folded_or_offset:
; CHECK: i64.load8_s $push{{[0-9]+}}=, 2($pop{{[0-9]+}}){{$}}
define i64 @load_i8_i64_s_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load i8, i8* %arrayidx
  %conv = sext i8 %t1 to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i16_i32_s_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load16_s  $push1=, 42($pop0){{$}}
define i32 @load_i16_i32_s_from_numeric_address() {
  %s = inttoptr i32 42 to i16*
  %t = load i16, i16* %s
  %u = sext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i8_i32_s_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load8_s  $push1=, gv8($pop0){{$}}
@gv8 = global i8 0
define i32 @load_i8_i32_s_from_global_address() {
  %t = load i8, i8* @gv8
  %u = sext i8 %t to i32
  ret i32 %u
}

;===----------------------------------------------------------------------------
; Zero-extending loads
;===----------------------------------------------------------------------------

; Fold an offset into a zero-extending load.

; CHECK-LABEL: load_i8_i32_z_with_folded_offset:
; CHECK: i32.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_i32_z_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load i8, i8* %s
  %u = zext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i32_i64_z_with_folded_offset:
; CHECK: i64.load32_u $push0=, 24($0){{$}}
define i64 @load_i32_i64_z_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load i32, i32* %s
  %u = zext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a zero-extending load.

; CHECK-LABEL: load_i8_i32_z_with_folded_gep_offset:
; CHECK: i32.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_i32_z_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load i8, i8* %s
  %u = zext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i32_z_with_folded_gep_offset:
; CHECK: i32.load16_u $push0=, 48($0){{$}}
define i32 @load_i16_i32_z_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = load i16, i16* %s
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i64_z_with_folded_gep_offset:
; CHECK: i64.load16_u $push0=, 48($0){{$}}
define i64 @load_i16_i64_z_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i64 24
  %t = load i16, i16* %s
  %u = zext i16 %t to i64
  ret i64 %u
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i16_i32_z_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load16_u  $push1=, 42($pop0){{$}}
define i32 @load_i16_i32_z_from_numeric_address() {
  %s = inttoptr i32 42 to i16*
  %t = load i16, i16* %s
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i8_i32_z_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load8_u  $push1=, gv8($pop0){{$}}
define i32 @load_i8_i32_z_from_global_address() {
  %t = load i8, i8* @gv8
  %u = zext i8 %t to i32
  ret i32 %u
}

; i8 return value should test anyext loads
; CHECK-LABEL: load_i8_i32_retvalue:
; CHECK: i32.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i8 @load_i8_i32_retvalue(i8 *%p) {
  %v = load i8, i8* %p
  ret i8 %v
}

;===----------------------------------------------------------------------------
; Truncating stores
;===----------------------------------------------------------------------------

; Fold an offset into a truncating store.

; CHECK-LABEL: store_i8_i32_with_folded_offset:
; CHECK: i32.store8 24($0), $1{{$}}
define void @store_i8_i32_with_folded_offset(i8* %p, i32 %v) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = trunc i32 %v to i8
  store i8 %t, i8* %s
  ret void
}

; CHECK-LABEL: store_i32_i64_with_folded_offset:
; CHECK: i64.store32 24($0), $1{{$}}
define void @store_i32_i64_with_folded_offset(i32* %p, i64 %v) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = trunc i64 %v to i32
  store i32 %t, i32* %s
  ret void
}

; Fold a gep offset into a truncating store.

; CHECK-LABEL: store_i8_i32_with_folded_gep_offset:
; CHECK: i32.store8 24($0), $1{{$}}
define void @store_i8_i32_with_folded_gep_offset(i8* %p, i32 %v) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = trunc i32 %v to i8
  store i8 %t, i8* %s
  ret void
}

; CHECK-LABEL: store_i16_i32_with_folded_gep_offset:
; CHECK: i32.store16 48($0), $1{{$}}
define void @store_i16_i32_with_folded_gep_offset(i16* %p, i32 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i32 %v to i16
  store i16 %t, i16* %s
  ret void
}

; CHECK-LABEL: store_i16_i64_with_folded_gep_offset:
; CHECK: i64.store16 48($0), $1{{$}}
define void @store_i16_i64_with_folded_gep_offset(i16* %p, i64 %v) {
  %s = getelementptr inbounds i16, i16* %p, i64 24
  %t = trunc i64 %v to i16
  store i16 %t, i16* %s
  ret void
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: store_i8_i32_with_folded_or_offset:
; CHECK: i32.store8 2($pop{{[0-9]+}}), $1{{$}}
define void @store_i8_i32_with_folded_or_offset(i32 %x, i32 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 2
  %t = trunc i32 %v to i8
  store i8 %t, i8* %arrayidx
  ret void
}

; CHECK-LABEL: store_i8_i64_with_folded_or_offset:
; CHECK: i64.store8 2($pop{{[0-9]+}}), $1{{$}}
define void @store_i8_i64_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 2
  %t = trunc i64 %v to i8
  store i8 %t, i8* %arrayidx
  ret void
}

;===----------------------------------------------------------------------------
; Aggregate values
;===----------------------------------------------------------------------------

; Fold the offsets when lowering aggregate loads and stores.

; CHECK-LABEL: aggregate_load_store:
; CHECK: i32.load  $2=, 0($0){{$}}
; CHECK: i32.load  $3=, 4($0){{$}}
; CHECK: i32.load  $4=, 8($0){{$}}
; CHECK: i32.load  $push0=, 12($0){{$}}
; CHECK: i32.store 12($1), $pop0{{$}}
; CHECK: i32.store 8($1), $4{{$}}
; CHECK: i32.store 4($1), $3{{$}}
; CHECK: i32.store 0($1), $2{{$}}
define void @aggregate_load_store({i32,i32,i32,i32}* %p, {i32,i32,i32,i32}* %q) {
  ; volatile so that things stay in order for the tests above
  %t = load volatile {i32,i32,i32,i32}, {i32, i32,i32,i32}* %p
  store volatile {i32,i32,i32,i32} %t, {i32, i32,i32,i32}* %q
  ret void
}

; Fold the offsets when lowering aggregate return values. The stores get
; merged into i64 stores.

; CHECK-LABEL: aggregate_return:
; CHECK: i64.const   $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK: i64.store   8($0), $pop[[L0]]{{$}}
; CHECK: i64.const   $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK: i64.store   0($0), $pop[[L1]]{{$}}
define {i32,i32,i32,i32} @aggregate_return() {
  ret {i32,i32,i32,i32} zeroinitializer
}

; Fold the offsets when lowering aggregate return values. The stores are not
; merged.

; CHECK-LABEL: aggregate_return_without_merge:
; CHECK: i32.const   $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK: i32.store8  14($0), $pop[[L0]]{{$}}
; CHECK: i32.const   $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK: i32.store16 12($0), $pop[[L1]]{{$}}
; CHECK: i32.const   $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK: i32.store   8($0), $pop[[L2]]{{$}}
; CHECK: i64.const   $push[[L3:[0-9]+]]=, 0{{$}}
; CHECK: i64.store   0($0), $pop[[L3]]{{$}}
define {i64,i32,i16,i8} @aggregate_return_without_merge() {
  ret {i64,i32,i16,i8} zeroinitializer
}
