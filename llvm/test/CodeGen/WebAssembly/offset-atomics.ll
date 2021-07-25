; RUN: not --crash llc > /dev/null < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics,+sign-ext | FileCheck %s

; Test that atomic loads are assembled properly.

target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Atomic loads: 32-bit
;===----------------------------------------------------------------------------

; Basic load.

; CHECK-LABEL: load_i32_no_offset:
; CHECK: i32.atomic.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_no_offset(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_offset:
; CHECK: i32.atomic.load $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_gep_offset:
; CHECK: i32.atomic.load $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.load $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.load $push2=, 0($pop1){{$}}
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
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.load $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load $push1=, 42($pop0){{$}}
define i32 @load_i32_from_numeric_address() {
  %s = inttoptr i32 42 to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  ret i32 %t
}

; CHECK-LABEL: load_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load $push1=, gv($pop0){{$}}
@gv = global i32 0
define i32 @load_i32_from_global_address() {
  %t = load atomic i32, i32* @gv seq_cst, align 4
  ret i32 %t
}

;===----------------------------------------------------------------------------
; Atomic loads: 64-bit
;===----------------------------------------------------------------------------

; Basic load.

; CHECK-LABEL: load_i64_no_offset:
; CHECK: i64.atomic.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @load_i64_no_offset(i64 *%p) {
  %v = load atomic i64, i64* %p seq_cst, align 8
  ret i64 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i64_with_folded_offset:
; CHECK: i64.atomic.load $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i64_with_folded_gep_offset:
; CHECK: i64.atomic.load $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.load $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.load $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: load_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.load $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  %t = load atomic i64, i64* %s seq_cst, align 8
  ret i64 %t
}

;===----------------------------------------------------------------------------
; Atomic stores: 32-bit
;===----------------------------------------------------------------------------

; Basic store.

; CHECK-LABEL: store_i32_no_offset:
; CHECK-NEXT: .functype store_i32_no_offset (i32, i32) -> (){{$}}
; CHECK-NEXT: i32.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i32_no_offset(i32 *%p, i32 %v) {
  store atomic i32 %v, i32* %p seq_cst, align 4
  ret void
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: store_i32_with_folded_offset:
; CHECK: i32.atomic.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: store_i32_with_folded_gep_offset:
; CHECK: i32.atomic.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: store_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: store_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: store_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; When storing from a fixed address, materialize a zero.

; CHECK-LABEL: store_i32_to_numeric_address:
; CHECK:      i32.const $push0=, 0{{$}}
; CHECK-NEXT: i32.const $push1=, 0{{$}}
; CHECK-NEXT: i32.atomic.store 42($pop0), $pop1{{$}}
define void @store_i32_to_numeric_address() {
  %s = inttoptr i32 42 to i32*
  store atomic i32 0, i32* %s seq_cst, align 4
  ret void
}

; CHECK-LABEL: store_i32_to_global_address:
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.const $push1=, 0{{$}}
; CHECK: i32.atomic.store gv($pop0), $pop1{{$}}
define void @store_i32_to_global_address() {
  store atomic i32 0, i32* @gv seq_cst, align 4
  ret void
}

;===----------------------------------------------------------------------------
; Atomic stores: 64-bit
;===----------------------------------------------------------------------------

; Basic store.

; CHECK-LABEL: store_i64_no_offset:
; CHECK-NEXT: .functype store_i64_no_offset (i32, i64) -> (){{$}}
; CHECK-NEXT: i64.atomic.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_i64_no_offset(i64 *%p, i64 %v) {
  store atomic i64 %v, i64* %p seq_cst, align 8
  ret void
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: store_i64_with_folded_offset:
; CHECK: i64.atomic.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: store_i64_with_folded_gep_offset:
; CHECK: i64.atomic.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: store_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: store_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: store_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  store atomic i64 0, i64* %s seq_cst, align 8
  ret void
}

;===----------------------------------------------------------------------------
; Atomic sign-extending loads
;===----------------------------------------------------------------------------

; Fold an offset into a sign-extending load.

; CHECK-LABEL: load_i8_i32_s_with_folded_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @load_i8_i32_s_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = sext i8 %t to i32
  ret i32 %u
}

; 32->64 sext load gets selected as i32.atomic.load, i64.extend_i32_s
; CHECK-LABEL: load_i32_i64_s_with_folded_offset:
; CHECK: i32.atomic.load $push0=, 24($0){{$}}
; CHECK-NEXT: i64.extend_i32_s $push1=, $pop0{{$}}
define i64 @load_i32_i64_s_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  %u = sext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a sign-extending load.

; CHECK-LABEL: load_i8_i32_s_with_folded_gep_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @load_i8_i32_s_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = sext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i32_s_with_folded_gep_offset:
; CHECK: i32.atomic.load16_u $push0=, 48($0){{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0
define i32 @load_i16_i32_s_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = sext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i64_s_with_folded_gep_offset:
; CHECK: i64.atomic.load16_u $push0=, 48($0){{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0
define i64 @load_i16_i64_s_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = sext i16 %t to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: load_i8_i32_s_with_folded_or_offset:
; CHECK: i32.atomic.load8_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}){{$}}
; CHECK-NEXT: i32.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i32 @load_i8_i32_s_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load atomic i8, i8* %arrayidx seq_cst, align 1
  %conv = sext i8 %t1 to i32
  ret i32 %conv
}

; CHECK-LABEL: load_i8_i64_s_with_folded_or_offset:
; CHECK: i64.atomic.load8_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}){{$}}
; CHECK-NEXT: i64.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i64 @load_i8_i64_s_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load atomic i8, i8* %arrayidx seq_cst, align 1
  %conv = sext i8 %t1 to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i16_i32_s_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load16_u $push1=, 42($pop0){{$}}
; CHECK-NEXT: i32.extend16_s $push2=, $pop1
define i32 @load_i16_i32_s_from_numeric_address() {
  %s = inttoptr i32 42 to i16*
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = sext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i8_i32_s_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load8_u $push1=, gv8($pop0){{$}}
; CHECK-NEXT: i32.extend8_s $push2=, $pop1{{$}}
@gv8 = global i8 0
define i32 @load_i8_i32_s_from_global_address() {
  %t = load atomic i8, i8* @gv8 seq_cst, align 1
  %u = sext i8 %t to i32
  ret i32 %u
}

;===----------------------------------------------------------------------------
; Atomic zero-extending loads
;===----------------------------------------------------------------------------

; Fold an offset into a zero-extending load.

; CHECK-LABEL: load_i8_i32_z_with_folded_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_i32_z_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = zext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i32_i64_z_with_folded_offset:
; CHECK: i64.atomic.load32_u $push0=, 24($0){{$}}
define i64 @load_i32_i64_z_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = load atomic i32, i32* %s seq_cst, align 4
  %u = zext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a zero-extending load.

; CHECK-LABEL: load_i8_i32_z_with_folded_gep_offset:
; CHECK: i32.atomic.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_i32_z_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load atomic i8, i8* %s seq_cst, align 1
  %u = zext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i32_z_with_folded_gep_offset:
; CHECK: i32.atomic.load16_u $push0=, 48($0){{$}}
define i32 @load_i16_i32_z_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i64_z_with_folded_gep_offset:
; CHECK: i64.atomic.load16_u $push0=, 48($0){{$}}
define i64 @load_i16_i64_z_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i64 24
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = zext i16 %t to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: load_i8_i32_z_with_folded_or_offset:
; CHECK: i32.atomic.load8_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}){{$}}
define i32 @load_i8_i32_z_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load atomic i8, i8* %arrayidx seq_cst, align 1
  %conv = zext i8 %t1 to i32
  ret i32 %conv
}

; CHECK-LABEL: load_i8_i64_z_with_folded_or_offset:
; CHECK: i64.atomic.load8_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}){{$}}
define i64 @load_i8_i64_z_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t1 = load atomic i8, i8* %arrayidx seq_cst, align 1
  %conv = zext i8 %t1 to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i16_i32_z_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load16_u $push1=, 42($pop0){{$}}
define i32 @load_i16_i32_z_from_numeric_address() {
  %s = inttoptr i32 42 to i16*
  %t = load atomic i16, i16* %s seq_cst, align 2
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i8_i32_z_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.load8_u $push1=, gv8($pop0){{$}}
define i32 @load_i8_i32_z_from_global_address() {
  %t = load atomic i8, i8* @gv8 seq_cst, align 1
  %u = zext i8 %t to i32
  ret i32 %u
}

; i8 return value should test anyext loads

; CHECK-LABEL: load_i8_i32_retvalue:
; CHECK: i32.atomic.load8_u $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i8 @load_i8_i32_retvalue(i8 *%p) {
  %v = load atomic i8, i8* %p seq_cst, align 1
  ret i8 %v
}

;===----------------------------------------------------------------------------
; Atomic truncating stores
;===----------------------------------------------------------------------------

; Fold an offset into a truncating store.

; CHECK-LABEL: store_i8_i32_with_folded_offset:
; CHECK: i32.atomic.store8 24($0), $1{{$}}
define void @store_i8_i32_with_folded_offset(i8* %p, i32 %v) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = trunc i32 %v to i8
  store atomic i8 %t, i8* %s seq_cst, align 1
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

; CHECK-LABEL: store_i8_i32_with_folded_gep_offset:
; CHECK: i32.atomic.store8 24($0), $1{{$}}
define void @store_i8_i32_with_folded_gep_offset(i8* %p, i32 %v) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = trunc i32 %v to i8
  store atomic i8 %t, i8* %s seq_cst, align 1
  ret void
}

; CHECK-LABEL: store_i16_i32_with_folded_gep_offset:
; CHECK: i32.atomic.store16 48($0), $1{{$}}
define void @store_i16_i32_with_folded_gep_offset(i16* %p, i32 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i32 %v to i16
  store atomic i16 %t, i16* %s seq_cst, align 2
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

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: store_i8_i32_with_folded_or_offset:
; CHECK: i32.atomic.store8 2($pop{{[0-9]+}}), $1{{$}}
define void @store_i8_i32_with_folded_or_offset(i32 %x, i32 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 2
  %t = trunc i32 %v to i8
  store atomic i8 %t, i8* %arrayidx seq_cst, align 1
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

;===----------------------------------------------------------------------------
; Atomic binary read-modify-writes: 32-bit
;===----------------------------------------------------------------------------

; There are several RMW instructions, but here we only test 'add' as an example.

; Basic RMW.

; CHECK-LABEL: rmw_add_i32_no_offset:
; CHECK-NEXT: .functype rmw_add_i32_no_offset (i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @rmw_add_i32_no_offset(i32* %p, i32 %v) {
  %old = atomicrmw add i32* %p, i32 %v seq_cst
  ret i32 %old
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: rmw_add_i32_with_folded_offset:
; CHECK: i32.atomic.rmw.add $push0=, 24($0), $1{{$}}
define i32 @rmw_add_i32_with_folded_offset(i32* %p, i32 %v) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %old = atomicrmw add i32* %s, i32 %v seq_cst
  ret i32 %old
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: rmw_add_i32_with_folded_gep_offset:
; CHECK: i32.atomic.rmw.add $push0=, 24($0), $1{{$}}
define i32 @rmw_add_i32_with_folded_gep_offset(i32* %p, i32 %v) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %old = atomicrmw add i32* %s, i32 %v seq_cst
  ret i32 %old
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: rmw_add_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.rmw.add $push2=, 0($pop1), $1{{$}}
define i32 @rmw_add_i32_with_unfolded_gep_negative_offset(i32* %p, i32 %v) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %old = atomicrmw add i32* %s, i32 %v seq_cst
  ret i32 %old
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: rmw_add_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.rmw.add $push2=, 0($pop1), $1{{$}}
define i32 @rmw_add_i32_with_unfolded_offset(i32* %p, i32 %v) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %old = atomicrmw add i32* %s, i32 %v seq_cst
  ret i32 %old
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: rmw_add_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.rmw.add $push2=, 0($pop1), $1{{$}}
define i32 @rmw_add_i32_with_unfolded_gep_offset(i32* %p, i32 %v) {
  %s = getelementptr i32, i32* %p, i32 6
  %old = atomicrmw add i32* %s, i32 %v seq_cst
  ret i32 %old
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: rmw_add_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw.add $push1=, 42($pop0), $0{{$}}
define i32 @rmw_add_i32_from_numeric_address(i32 %v) {
  %s = inttoptr i32 42 to i32*
  %old = atomicrmw add i32* %s, i32 %v seq_cst
  ret i32 %old
}

; CHECK-LABEL: rmw_add_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw.add $push1=, gv($pop0), $0{{$}}
define i32 @rmw_add_i32_from_global_address(i32 %v) {
  %old = atomicrmw add i32* @gv, i32 %v seq_cst
  ret i32 %old
}

;===----------------------------------------------------------------------------
; Atomic binary read-modify-writes: 64-bit
;===----------------------------------------------------------------------------

; Basic RMW.

; CHECK-LABEL: rmw_add_i64_no_offset:
; CHECK-NEXT: .functype rmw_add_i64_no_offset (i32, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.add $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @rmw_add_i64_no_offset(i64* %p, i64 %v) {
  %old = atomicrmw add i64* %p, i64 %v seq_cst
  ret i64 %old
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: rmw_add_i64_with_folded_offset:
; CHECK: i64.atomic.rmw.add $push0=, 24($0), $1{{$}}
define i64 @rmw_add_i64_with_folded_offset(i64* %p, i64 %v) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %old = atomicrmw add i64* %s, i64 %v seq_cst
  ret i64 %old
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: rmw_add_i64_with_folded_gep_offset:
; CHECK: i64.atomic.rmw.add $push0=, 24($0), $1{{$}}
define i64 @rmw_add_i64_with_folded_gep_offset(i64* %p, i64 %v) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %old = atomicrmw add i64* %s, i64 %v seq_cst
  ret i64 %old
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: rmw_add_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.rmw.add $push2=, 0($pop1), $1{{$}}
define i64 @rmw_add_i64_with_unfolded_gep_negative_offset(i64* %p, i64 %v) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %old = atomicrmw add i64* %s, i64 %v seq_cst
  ret i64 %old
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: rmw_add_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.rmw.add $push2=, 0($pop1), $1{{$}}
define i64 @rmw_add_i64_with_unfolded_offset(i64* %p, i64 %v) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %old = atomicrmw add i64* %s, i64 %v seq_cst
  ret i64 %old
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: rmw_add_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.rmw.add $push2=, 0($pop1), $1{{$}}
define i64 @rmw_add_i64_with_unfolded_gep_offset(i64* %p, i64 %v) {
  %s = getelementptr i64, i64* %p, i32 3
  %old = atomicrmw add i64* %s, i64 %v seq_cst
  ret i64 %old
}

;===----------------------------------------------------------------------------
; Atomic truncating & sign-extending binary RMWs
;===----------------------------------------------------------------------------

; Fold an offset into a sign-extending rmw.

; CHECK-LABEL: rmw_add_i8_i32_s_with_folded_offset:
; CHECK: i32.atomic.rmw8.add_u $push0=, 24($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @rmw_add_i8_i32_s_with_folded_offset(i8* %p, i32 %v) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %s, i8 %t seq_cst
  %u = sext i8 %old to i32
  ret i32 %u
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.add, i64.extend_i32_s
; CHECK-LABEL: rmw_add_i32_i64_s_with_folded_offset:
; CHECK: i32.wrap_i64 $push0=, $1
; CHECK-NEXT: i32.atomic.rmw.add $push1=, 24($0), $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push2=, $pop1{{$}}
define i64 @rmw_add_i32_i64_s_with_folded_offset(i32* %p, i64 %v) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = trunc i64 %v to i32
  %old = atomicrmw add i32* %s, i32 %t seq_cst
  %u = sext i32 %old to i64
  ret i64 %u
}

; Fold a gep offset into a sign-extending rmw.

; CHECK-LABEL: rmw_add_i8_i32_s_with_folded_gep_offset:
; CHECK: i32.atomic.rmw8.add_u $push0=, 24($0), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @rmw_add_i8_i32_s_with_folded_gep_offset(i8* %p, i32 %v) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %s, i8 %t seq_cst
  %u = sext i8 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i16_i32_s_with_folded_gep_offset:
; CHECK: i32.atomic.rmw16.add_u $push0=, 48($0), $1{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0
define i32 @rmw_add_i16_i32_s_with_folded_gep_offset(i16* %p, i32 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %s, i16 %t seq_cst
  %u = sext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i16_i64_s_with_folded_gep_offset:
; CHECK: i64.atomic.rmw16.add_u $push0=, 48($0), $1{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0
define i64 @rmw_add_i16_i64_s_with_folded_gep_offset(i16* %p, i64 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i64 %v to i16
  %old = atomicrmw add i16* %s, i16 %t seq_cst
  %u = sext i16 %old to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: rmw_add_i8_i32_s_with_folded_or_offset:
; CHECK: i32.atomic.rmw8.add_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1{{$}}
; CHECK-NEXT: i32.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i32 @rmw_add_i8_i32_s_with_folded_or_offset(i32 %x, i32 %v) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %arrayidx, i8 %t seq_cst
  %conv = sext i8 %old to i32
  ret i32 %conv
}

; CHECK-LABEL: rmw_add_i8_i64_s_with_folded_or_offset:
; CHECK: i64.atomic.rmw8.add_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1{{$}}
; CHECK-NEXT: i64.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i64 @rmw_add_i8_i64_s_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t = trunc i64 %v to i8
  %old = atomicrmw add i8* %arrayidx, i8 %t seq_cst
  %conv = sext i8 %old to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: rmw_add_i16_i32_s_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw16.add_u $push1=, 42($pop0), $0{{$}}
; CHECK-NEXT: i32.extend16_s $push2=, $pop1
define i32 @rmw_add_i16_i32_s_from_numeric_address(i32 %v) {
  %s = inttoptr i32 42 to i16*
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %s, i16 %t seq_cst
  %u = sext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i8_i32_s_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw8.add_u $push1=, gv8($pop0), $0{{$}}
; CHECK-NEXT: i32.extend8_s $push2=, $pop1{{$}}
define i32 @rmw_add_i8_i32_s_from_global_address(i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* @gv8, i8 %t seq_cst
  %u = sext i8 %old to i32
  ret i32 %u
}

;===----------------------------------------------------------------------------
; Atomic truncating & zero-extending binary RMWs
;===----------------------------------------------------------------------------

; Fold an offset into a zero-extending rmw.

; CHECK-LABEL: rmw_add_i8_i32_z_with_folded_offset:
; CHECK: i32.atomic.rmw8.add_u $push0=, 24($0), $1{{$}}
define i32 @rmw_add_i8_i32_z_with_folded_offset(i8* %p, i32 %v) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %s, i8 %t seq_cst
  %u = zext i8 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i32_i64_z_with_folded_offset:
; CHECK: i64.atomic.rmw32.add_u $push0=, 24($0), $1{{$}}
define i64 @rmw_add_i32_i64_z_with_folded_offset(i32* %p, i64 %v) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = trunc i64 %v to i32
  %old = atomicrmw add i32* %s, i32 %t seq_cst
  %u = zext i32 %old to i64
  ret i64 %u
}

; Fold a gep offset into a zero-extending rmw.

; CHECK-LABEL: rmw_add_i8_i32_z_with_folded_gep_offset:
; CHECK: i32.atomic.rmw8.add_u $push0=, 24($0), $1{{$}}
define i32 @rmw_add_i8_i32_z_with_folded_gep_offset(i8* %p, i32 %v) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %s, i8 %t seq_cst
  %u = zext i8 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i16_i32_z_with_folded_gep_offset:
; CHECK: i32.atomic.rmw16.add_u $push0=, 48($0), $1{{$}}
define i32 @rmw_add_i16_i32_z_with_folded_gep_offset(i16* %p, i32 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %s, i16 %t seq_cst
  %u = zext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i16_i64_z_with_folded_gep_offset:
; CHECK: i64.atomic.rmw16.add_u $push0=, 48($0), $1{{$}}
define i64 @rmw_add_i16_i64_z_with_folded_gep_offset(i16* %p, i64 %v) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %t = trunc i64 %v to i16
  %old = atomicrmw add i16* %s, i16 %t seq_cst
  %u = zext i16 %old to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: rmw_add_i8_i32_z_with_folded_or_offset:
; CHECK: i32.atomic.rmw8.add_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1{{$}}
define i32 @rmw_add_i8_i32_z_with_folded_or_offset(i32 %x, i32 %v) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %arrayidx, i8 %t seq_cst
  %conv = zext i8 %old to i32
  ret i32 %conv
}

; CHECK-LABEL: rmw_add_i8_i64_z_with_folded_or_offset:
; CHECK: i64.atomic.rmw8.add_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1{{$}}
define i64 @rmw_add_i8_i64_z_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %t = trunc i64 %v to i8
  %old = atomicrmw add i8* %arrayidx, i8 %t seq_cst
  %conv = zext i8 %old to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: rmw_add_i16_i32_z_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw16.add_u $push1=, 42($pop0), $0{{$}}
define i32 @rmw_add_i16_i32_z_from_numeric_address(i32 %v) {
  %s = inttoptr i32 42 to i16*
  %t = trunc i32 %v to i16
  %old = atomicrmw add i16* %s, i16 %t seq_cst
  %u = zext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: rmw_add_i8_i32_z_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw8.add_u $push1=, gv8($pop0), $0{{$}}
define i32 @rmw_add_i8_i32_z_from_global_address(i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* @gv8, i8 %t seq_cst
  %u = zext i8 %old to i32
  ret i32 %u
}

; i8 return value should test anyext RMWs

; CHECK-LABEL: rmw_add_i8_i32_retvalue:
; CHECK: i32.atomic.rmw8.add_u $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i8 @rmw_add_i8_i32_retvalue(i8 *%p, i32 %v) {
  %t = trunc i32 %v to i8
  %old = atomicrmw add i8* %p, i8 %t seq_cst
  ret i8 %old
}

;===----------------------------------------------------------------------------
; Atomic ternary read-modify-writes: 32-bit
;===----------------------------------------------------------------------------

; Basic RMW.

; CHECK-LABEL: cmpxchg_i32_no_offset:
; CHECK-NEXT: .functype cmpxchg_i32_no_offset (i32, i32, i32) -> (i32){{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @cmpxchg_i32_no_offset(i32* %p, i32 %exp, i32 %new) {
  %pair = cmpxchg i32* %p, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: cmpxchg_i32_with_folded_offset:
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 24($0), $1, $2{{$}}
define i32 @cmpxchg_i32_with_folded_offset(i32* %p, i32 %exp, i32 %new) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %pair = cmpxchg i32* %s, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: cmpxchg_i32_with_folded_gep_offset:
; CHECK: i32.atomic.rmw.cmpxchg $push0=, 24($0), $1, $2{{$}}
define i32 @cmpxchg_i32_with_folded_gep_offset(i32* %p, i32 %exp, i32 %new) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %pair = cmpxchg i32* %s, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: cmpxchg_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push2=, 0($pop1), $1, $2{{$}}
define i32 @cmpxchg_i32_with_unfolded_gep_negative_offset(i32* %p, i32 %exp, i32 %new) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %pair = cmpxchg i32* %s, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: cmpxchg_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push2=, 0($pop1), $1, $2{{$}}
define i32 @cmpxchg_i32_with_unfolded_offset(i32* %p, i32 %exp, i32 %new) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %pair = cmpxchg i32* %s, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: cmpxchg_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push2=, 0($pop1), $1, $2{{$}}
define i32 @cmpxchg_i32_with_unfolded_gep_offset(i32* %p, i32 %exp, i32 %new) {
  %s = getelementptr i32, i32* %p, i32 6
  %pair = cmpxchg i32* %s, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: cmpxchg_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push1=, 42($pop0), $0, $1{{$}}
define i32 @cmpxchg_i32_from_numeric_address(i32 %exp, i32 %new) {
  %s = inttoptr i32 42 to i32*
  %pair = cmpxchg i32* %s, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

; CHECK-LABEL: cmpxchg_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw.cmpxchg $push1=, gv($pop0), $0, $1{{$}}
define i32 @cmpxchg_i32_from_global_address(i32 %exp, i32 %new) {
  %pair = cmpxchg i32* @gv, i32 %exp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

;===----------------------------------------------------------------------------
; Atomic ternary read-modify-writes: 64-bit
;===----------------------------------------------------------------------------

; Basic RMW.

; CHECK-LABEL: cmpxchg_i64_no_offset:
; CHECK-NEXT: .functype cmpxchg_i64_no_offset (i32, i64, i64) -> (i64){{$}}
; CHECK: i64.atomic.rmw.cmpxchg $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @cmpxchg_i64_no_offset(i64* %p, i64 %exp, i64 %new) {
  %pair = cmpxchg i64* %p, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: cmpxchg_i64_with_folded_offset:
; CHECK: i64.atomic.rmw.cmpxchg $push0=, 24($0), $1, $2{{$}}
define i64 @cmpxchg_i64_with_folded_offset(i64* %p, i64 %exp, i64 %new) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %pair = cmpxchg i64* %s, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: cmpxchg_i64_with_folded_gep_offset:
; CHECK: i64.atomic.rmw.cmpxchg $push0=, 24($0), $1, $2{{$}}
define i64 @cmpxchg_i64_with_folded_gep_offset(i64* %p, i64 %exp, i64 %new) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %pair = cmpxchg i64* %s, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: cmpxchg_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.rmw.cmpxchg $push2=, 0($pop1), $1, $2{{$}}
define i64 @cmpxchg_i64_with_unfolded_gep_negative_offset(i64* %p, i64 %exp, i64 %new) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %pair = cmpxchg i64* %s, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: cmpxchg_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.rmw.cmpxchg $push2=, 0($pop1), $1, $2{{$}}
define i64 @cmpxchg_i64_with_unfolded_offset(i64* %p, i64 %exp, i64 %new) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %pair = cmpxchg i64* %s, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: cmpxchg_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: i64.atomic.rmw.cmpxchg $push2=, 0($pop1), $1, $2{{$}}
define i64 @cmpxchg_i64_with_unfolded_gep_offset(i64* %p, i64 %exp, i64 %new) {
  %s = getelementptr i64, i64* %p, i32 3
  %pair = cmpxchg i64* %s, i64 %exp, i64 %new seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

;===----------------------------------------------------------------------------
; Atomic truncating & sign-extending ternary RMWs
;===----------------------------------------------------------------------------

; Fold an offset into a sign-extending rmw.

; CHECK-LABEL: cmpxchg_i8_i32_s_with_folded_offset:
; CHECK: i32.atomic.rmw8.cmpxchg_u $push0=, 24($0), $1, $2{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @cmpxchg_i8_i32_s_with_folded_offset(i8* %p, i32 %exp, i32 %new) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %s, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %u = sext i8 %old to i32
  ret i32 %u
}

; 32->64 sext rmw gets selected as i32.atomic.rmw.cmpxchg, i64.extend_i32_s
; CHECK-LABEL: cmpxchg_i32_i64_s_with_folded_offset:
; CHECK: i32.wrap_i64 $push1=, $1
; CHECK-NEXT: i32.wrap_i64 $push0=, $2
; CHECK-NEXT: i32.atomic.rmw.cmpxchg $push2=, 24($0), $pop1, $pop0{{$}}
; CHECK-NEXT: i64.extend_i32_s $push3=, $pop2{{$}}
define i64 @cmpxchg_i32_i64_s_with_folded_offset(i32* %p, i64 %exp, i64 %new) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %exp_t = trunc i64 %exp to i32
  %new_t = trunc i64 %new to i32
  %pair = cmpxchg i32* %s, i32 %exp_t, i32 %new_t seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  %u = sext i32 %old to i64
  ret i64 %u
}

; Fold a gep offset into a sign-extending rmw.

; CHECK-LABEL: cmpxchg_i8_i32_s_with_folded_gep_offset:
; CHECK: i32.atomic.rmw8.cmpxchg_u $push0=, 24($0), $1, $2{{$}}
; CHECK-NEXT: i32.extend8_s $push1=, $pop0
define i32 @cmpxchg_i8_i32_s_with_folded_gep_offset(i8* %p, i32 %exp, i32 %new) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %s, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %u = sext i8 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i16_i32_s_with_folded_gep_offset:
; CHECK: i32.atomic.rmw16.cmpxchg_u $push0=, 48($0), $1, $2{{$}}
; CHECK-NEXT: i32.extend16_s $push1=, $pop0
define i32 @cmpxchg_i16_i32_s_with_folded_gep_offset(i16* %p, i32 %exp, i32 %new) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %exp_t = trunc i32 %exp to i16
  %new_t = trunc i32 %new to i16
  %pair = cmpxchg i16* %s, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %u = sext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i16_i64_s_with_folded_gep_offset:
; CHECK: i64.atomic.rmw16.cmpxchg_u $push0=, 48($0), $1, $2{{$}}
; CHECK-NEXT: i64.extend16_s $push1=, $pop0
define i64 @cmpxchg_i16_i64_s_with_folded_gep_offset(i16* %p, i64 %exp, i64 %new) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %exp_t = trunc i64 %exp to i16
  %new_t = trunc i64 %new to i16
  %pair = cmpxchg i16* %s, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %u = sext i16 %old to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: cmpxchg_i8_i32_s_with_folded_or_offset:
; CHECK: i32.atomic.rmw8.cmpxchg_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1, $2{{$}}
; CHECK-NEXT: i32.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i32 @cmpxchg_i8_i32_s_with_folded_or_offset(i32 %x, i32 %exp, i32 %new) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %arrayidx, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %conv = sext i8 %old to i32
  ret i32 %conv
}

; CHECK-LABEL: cmpxchg_i8_i64_s_with_folded_or_offset:
; CHECK: i64.atomic.rmw8.cmpxchg_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1, $2{{$}}
; CHECK-NEXT: i64.extend8_s $push{{[0-9]+}}=, $pop[[R1]]{{$}}
define i64 @cmpxchg_i8_i64_s_with_folded_or_offset(i32 %x, i64 %exp, i64 %new) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %exp_t = trunc i64 %exp to i8
  %new_t = trunc i64 %new to i8
  %pair = cmpxchg i8* %arrayidx, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %conv = sext i8 %old to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: cmpxchg_i16_i32_s_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw16.cmpxchg_u $push1=, 42($pop0), $0, $1{{$}}
; CHECK-NEXT: i32.extend16_s $push2=, $pop1
define i32 @cmpxchg_i16_i32_s_from_numeric_address(i32 %exp, i32 %new) {
  %s = inttoptr i32 42 to i16*
  %exp_t = trunc i32 %exp to i16
  %new_t = trunc i32 %new to i16
  %pair = cmpxchg i16* %s, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %u = sext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i8_i32_s_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw8.cmpxchg_u $push1=, gv8($pop0), $0, $1{{$}}
; CHECK-NEXT: i32.extend8_s $push2=, $pop1{{$}}
define i32 @cmpxchg_i8_i32_s_from_global_address(i32 %exp, i32 %new) {
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* @gv8, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %u = sext i8 %old to i32
  ret i32 %u
}

;===----------------------------------------------------------------------------
; Atomic truncating & zero-extending ternary RMWs
;===----------------------------------------------------------------------------

; Fold an offset into a sign-extending rmw.

; CHECK-LABEL: cmpxchg_i8_i32_z_with_folded_offset:
; CHECK: i32.atomic.rmw8.cmpxchg_u $push0=, 24($0), $1, $2{{$}}
define i32 @cmpxchg_i8_i32_z_with_folded_offset(i8* %p, i32 %exp, i32 %new) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %s, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %u = zext i8 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i32_i64_z_with_folded_offset:
; CHECK: i64.atomic.rmw32.cmpxchg_u $push0=, 24($0), $1, $2{{$}}
define i64 @cmpxchg_i32_i64_z_with_folded_offset(i32* %p, i64 %exp, i64 %new) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %exp_t = trunc i64 %exp to i32
  %new_t = trunc i64 %new to i32
  %pair = cmpxchg i32* %s, i32 %exp_t, i32 %new_t seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  %u = zext i32 %old to i64
  ret i64 %u
}

; Fold a gep offset into a sign-extending rmw.

; CHECK-LABEL: cmpxchg_i8_i32_z_with_folded_gep_offset:
; CHECK: i32.atomic.rmw8.cmpxchg_u $push0=, 24($0), $1, $2{{$}}
define i32 @cmpxchg_i8_i32_z_with_folded_gep_offset(i8* %p, i32 %exp, i32 %new) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %s, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %u = zext i8 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i16_i32_z_with_folded_gep_offset:
; CHECK: i32.atomic.rmw16.cmpxchg_u $push0=, 48($0), $1, $2{{$}}
define i32 @cmpxchg_i16_i32_z_with_folded_gep_offset(i16* %p, i32 %exp, i32 %new) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %exp_t = trunc i32 %exp to i16
  %new_t = trunc i32 %new to i16
  %pair = cmpxchg i16* %s, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %u = zext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i16_i64_z_with_folded_gep_offset:
; CHECK: i64.atomic.rmw16.cmpxchg_u $push0=, 48($0), $1, $2{{$}}
define i64 @cmpxchg_i16_i64_z_with_folded_gep_offset(i16* %p, i64 %exp, i64 %new) {
  %s = getelementptr inbounds i16, i16* %p, i32 24
  %exp_t = trunc i64 %exp to i16
  %new_t = trunc i64 %new to i16
  %pair = cmpxchg i16* %s, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %u = zext i16 %old to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: cmpxchg_i8_i32_z_with_folded_or_offset:
; CHECK: i32.atomic.rmw8.cmpxchg_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1, $2{{$}}
define i32 @cmpxchg_i8_i32_z_with_folded_or_offset(i32 %x, i32 %exp, i32 %new) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* %arrayidx, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %conv = zext i8 %old to i32
  ret i32 %conv
}

; CHECK-LABEL: cmpxchg_i8_i64_z_with_folded_or_offset:
; CHECK: i64.atomic.rmw8.cmpxchg_u $push[[R1:[0-9]+]]=, 2($pop{{[0-9]+}}), $1, $2{{$}}
define i64 @cmpxchg_i8_i64_z_with_folded_or_offset(i32 %x, i64 %exp, i64 %new) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to i8*
  %arrayidx = getelementptr inbounds i8, i8* %t0, i32 2
  %exp_t = trunc i64 %exp to i8
  %new_t = trunc i64 %new to i8
  %pair = cmpxchg i8* %arrayidx, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %conv = zext i8 %old to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: cmpxchg_i16_i32_z_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw16.cmpxchg_u $push1=, 42($pop0), $0, $1{{$}}
define i32 @cmpxchg_i16_i32_z_from_numeric_address(i32 %exp, i32 %new) {
  %s = inttoptr i32 42 to i16*
  %exp_t = trunc i32 %exp to i16
  %new_t = trunc i32 %new to i16
  %pair = cmpxchg i16* %s, i16 %exp_t, i16 %new_t seq_cst seq_cst
  %old = extractvalue { i16, i1 } %pair, 0
  %u = zext i16 %old to i32
  ret i32 %u
}

; CHECK-LABEL: cmpxchg_i8_i32_z_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.atomic.rmw8.cmpxchg_u $push1=, gv8($pop0), $0, $1{{$}}
define i32 @cmpxchg_i8_i32_z_from_global_address(i32 %exp, i32 %new) {
  %exp_t = trunc i32 %exp to i8
  %new_t = trunc i32 %new to i8
  %pair = cmpxchg i8* @gv8, i8 %exp_t, i8 %new_t seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pair, 0
  %u = zext i8 %old to i32
  ret i32 %u
}

;===----------------------------------------------------------------------------
; Waits: 32-bit
;===----------------------------------------------------------------------------

declare i32 @llvm.wasm.memory.atomic.wait32(i32*, i32, i64)

; Basic wait.

; CHECK-LABEL: wait32_no_offset:
; CHECK: memory.atomic.wait32 $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @wait32_no_offset(i32* %p, i32 %exp, i64 %timeout) {
  %v = call i32 @llvm.wasm.memory.atomic.wait32(i32* %p, i32 %exp, i64 %timeout)
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: wait32_with_folded_offset:
; CHECK: memory.atomic.wait32 $push0=, 24($0), $1, $2{{$}}
define i32 @wait32_with_folded_offset(i32* %p, i32 %exp, i64 %timeout) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* %s, i32 %exp, i64 %timeout)
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: wait32_with_folded_gep_offset:
; CHECK: memory.atomic.wait32 $push0=, 24($0), $1, $2{{$}}
define i32 @wait32_with_folded_gep_offset(i32* %p, i32 %exp, i64 %timeout) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* %s, i32 %exp, i64 %timeout)
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: wait32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.wait32 $push2=, 0($pop1), $1, $2{{$}}
define i32 @wait32_with_unfolded_gep_negative_offset(i32* %p, i32 %exp, i64 %timeout) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* %s, i32 %exp, i64 %timeout)
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: wait32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.wait32 $push2=, 0($pop1), $1, $2{{$}}
define i32 @wait32_with_unfolded_offset(i32* %p, i32 %exp, i64 %timeout) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* %s, i32 %exp, i64 %timeout)
  ret i32 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: wait32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.wait32 $push2=, 0($pop1), $1, $2{{$}}
define i32 @wait32_with_unfolded_gep_offset(i32* %p, i32 %exp, i64 %timeout) {
  %s = getelementptr i32, i32* %p, i32 6
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* %s, i32 %exp, i64 %timeout)
  ret i32 %t
}

; When waiting from a fixed address, materialize a zero.

; CHECK-LABEL: wait32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: memory.atomic.wait32 $push1=, 42($pop0), $0, $1{{$}}
define i32 @wait32_from_numeric_address(i32 %exp, i64 %timeout) {
  %s = inttoptr i32 42 to i32*
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* %s, i32 %exp, i64 %timeout)
  ret i32 %t
}

; CHECK-LABEL: wait32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: memory.atomic.wait32 $push1=, gv($pop0), $0, $1{{$}}
define i32 @wait32_from_global_address(i32 %exp, i64 %timeout) {
  %t = call i32 @llvm.wasm.memory.atomic.wait32(i32* @gv, i32 %exp, i64 %timeout)
  ret i32 %t
}

;===----------------------------------------------------------------------------
; Waits: 64-bit
;===----------------------------------------------------------------------------

declare i32 @llvm.wasm.memory.atomic.wait64(i64*, i64, i64)

; Basic wait.

; CHECK-LABEL: wait64_no_offset:
; CHECK: memory.atomic.wait64 $push0=, 0($0), $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @wait64_no_offset(i64* %p, i64 %exp, i64 %timeout) {
  %v = call i32 @llvm.wasm.memory.atomic.wait64(i64* %p, i64 %exp, i64 %timeout)
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: wait64_with_folded_offset:
; CHECK: memory.atomic.wait64 $push0=, 24($0), $1, $2{{$}}
define i32 @wait64_with_folded_offset(i64* %p, i64 %exp, i64 %timeout) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = call i32 @llvm.wasm.memory.atomic.wait64(i64* %s, i64 %exp, i64 %timeout)
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: wait64_with_folded_gep_offset:
; CHECK: memory.atomic.wait64 $push0=, 24($0), $1, $2{{$}}
define i32 @wait64_with_folded_gep_offset(i64* %p, i64 %exp, i64 %timeout) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %t = call i32 @llvm.wasm.memory.atomic.wait64(i64* %s, i64 %exp, i64 %timeout)
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: wait64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.wait64 $push2=, 0($pop1), $1, $2{{$}}
define i32 @wait64_with_unfolded_gep_negative_offset(i64* %p, i64 %exp, i64 %timeout) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %t = call i32 @llvm.wasm.memory.atomic.wait64(i64* %s, i64 %exp, i64 %timeout)
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: wait64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.wait64 $push2=, 0($pop1), $1, $2{{$}}
define i32 @wait64_with_unfolded_offset(i64* %p, i64 %exp, i64 %timeout) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = call i32 @llvm.wasm.memory.atomic.wait64(i64* %s, i64 %exp, i64 %timeout)
  ret i32 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: wait64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.wait64 $push2=, 0($pop1), $1, $2{{$}}
define i32 @wait64_with_unfolded_gep_offset(i64* %p, i64 %exp, i64 %timeout) {
  %s = getelementptr i64, i64* %p, i32 3
  %t = call i32 @llvm.wasm.memory.atomic.wait64(i64* %s, i64 %exp, i64 %timeout)
  ret i32 %t
}

;===----------------------------------------------------------------------------
; Notifies
;===----------------------------------------------------------------------------

declare i32 @llvm.wasm.memory.atomic.notify(i32*, i32)

; Basic notify.

; CHECK-LABEL: notify_no_offset:
; CHECK: memory.atomic.notify $push0=, 0($0), $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @notify_no_offset(i32* %p, i32 %notify_count) {
  %v = call i32 @llvm.wasm.memory.atomic.notify(i32* %p, i32 %notify_count)
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: notify_with_folded_offset:
; CHECK: memory.atomic.notify $push0=, 24($0), $1{{$}}
define i32 @notify_with_folded_offset(i32* %p, i32 %notify_count) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* %s, i32 %notify_count)
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: notify_with_folded_gep_offset:
; CHECK: memory.atomic.notify $push0=, 24($0), $1{{$}}
define i32 @notify_with_folded_gep_offset(i32* %p, i32 %notify_count) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* %s, i32 %notify_count)
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: notify_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.notify $push2=, 0($pop1), $1{{$}}
define i32 @notify_with_unfolded_gep_negative_offset(i32* %p, i32 %notify_count) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* %s, i32 %notify_count)
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: notify_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.notify $push2=, 0($pop1), $1{{$}}
define i32 @notify_with_unfolded_offset(i32* %p, i32 %notify_count) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* %s, i32 %notify_count)
  ret i32 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: notify_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add $push1=, $0, $pop0{{$}}
; CHECK: memory.atomic.notify $push2=, 0($pop1), $1{{$}}
define i32 @notify_with_unfolded_gep_offset(i32* %p, i32 %notify_count) {
  %s = getelementptr i32, i32* %p, i32 6
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* %s, i32 %notify_count)
  ret i32 %t
}

; When notifying from a fixed address, materialize a zero.

; CHECK-LABEL: notify_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: memory.atomic.notify $push1=, 42($pop0), $0{{$}}
define i32 @notify_from_numeric_address(i32 %notify_count) {
  %s = inttoptr i32 42 to i32*
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* %s, i32 %notify_count)
  ret i32 %t
}

; CHECK-LABEL: notify_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: memory.atomic.notify $push1=, gv($pop0), $0{{$}}
define i32 @notify_from_global_address(i32 %notify_count) {
  %t = call i32 @llvm.wasm.memory.atomic.notify(i32* @gv, i32 %notify_count)
  ret i32 %t
}
