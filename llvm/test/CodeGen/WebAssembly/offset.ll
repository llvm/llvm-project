; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers -disable-wasm-fallthrough-return-opt -mattr=+half-precision | FileCheck %s

; Test constant load and store address offsets.

target triple = "wasm32-unknown-unknown"

;===----------------------------------------------------------------------------
; Loads: 32-bit
;===----------------------------------------------------------------------------

; Basic load.

; CHECK-LABEL: load_i32_no_offset:
; CHECK: i32.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @load_i32_no_offset(ptr %p) {
  %v = load i32, ptr %p
  ret i32 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_offset:
; CHECK: i32.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i32, ptr %s
  ret i32 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i32_with_folded_gep_offset:
; CHECK: i32.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i32, ptr %p, i32 6
  %t = load i32, ptr %s
  ret i32 %t
}

; Same for nusw.

; CHECK-LABEL: load_i32_with_folded_gep_offset_nusw:
; CHECK: i32.load  $push0=, 24($0){{$}}
define i32 @load_i32_with_folded_gep_offset_nusw(ptr %p) {
  %s = getelementptr nusw i32, ptr %p, i32 6
  %t = load i32, ptr %s
  ret i32 %t
}

; For nuw we don't need the offset to be positive.

; CHECK-LABEL: load_i32_with_folded_gep_offset_nuw:
; CHECK: i32.load  $push0=, -24($0){{$}}
define i32 @load_i32_with_folded_gep_offset_nuw(ptr %p) {
  %s = getelementptr nuw i32, ptr %p, i32 -6
  %t = load i32, ptr %s
  ret i32 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_negative_offset(ptr %p) {
  %s = getelementptr inbounds i32, ptr %p, i32 -6
  %t = load i32, ptr %s
  ret i32 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i32, ptr %s
  ret i32 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: load_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.load  $push2=, 0($pop1){{$}}
define i32 @load_i32_with_unfolded_gep_offset(ptr %p) {
  %s = getelementptr i32, ptr %p, i32 6
  %t = load i32, ptr %s
  ret i32 %t
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i32_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load  $push1=, 42($pop0){{$}}
define i32 @load_i32_from_numeric_address() {
  %s = inttoptr i32 42 to ptr
  %t = load i32, ptr %s
  ret i32 %t
}

; CHECK-LABEL: load_i32_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load  $push1=, gv($pop0){{$}}
@gv = global i32 0
define i32 @load_i32_from_global_address() {
  %t = load i32, ptr @gv
  ret i32 %t
}

;===----------------------------------------------------------------------------
; Loads: 64-bit
;===----------------------------------------------------------------------------

; Basic load.

; CHECK-LABEL: load_i64_no_offset:
; CHECK: i64.load $push0=, 0($0){{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @load_i64_no_offset(ptr %p) {
  %v = load i64, ptr %p
  ret i64 %v
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: load_i64_with_folded_offset:
; CHECK: i64.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i64, ptr %s
  ret i64 %t
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: load_i64_with_folded_gep_offset:
; CHECK: i64.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i64, ptr %p, i32 3
  %t = load i64, ptr %s
  ret i64 %t
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: load_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_negative_offset(ptr %p) {
  %s = getelementptr inbounds i64, ptr %p, i32 -3
  %t = load i64, ptr %s
  ret i64 %t
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: load_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i64, ptr %s
  ret i64 %t
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: load_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_offset(ptr %p) {
  %s = getelementptr i64, ptr %p, i32 3
  %t = load i64, ptr %s
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
define void @store_i32_no_offset(ptr %p, i32 %v) {
  store i32 %v, ptr %p
  ret void
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: store_i32_with_folded_offset:
; CHECK: i32.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  store i32 0, ptr %s
  ret void
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: store_i32_with_folded_gep_offset:
; CHECK: i32.store 24($0), $pop0{{$}}
define void @store_i32_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i32, ptr %p, i32 6
  store i32 0, ptr %s
  ret void
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: store_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_negative_offset(ptr %p) {
  %s = getelementptr inbounds i32, ptr %p, i32 -6
  store i32 0, ptr %s
  ret void
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: store_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  store i32 0, ptr %s
  ret void
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: store_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_offset(ptr %p) {
  %s = getelementptr i32, ptr %p, i32 6
  store i32 0, ptr %s
  ret void
}

; When storing from a fixed address, materialize a zero.

; CHECK-LABEL: store_i32_to_numeric_address:
; CHECK:      i32.const $push0=, 0{{$}}
; CHECK-NEXT: i32.const $push1=, 0{{$}}
; CHECK-NEXT: i32.store 42($pop0), $pop1{{$}}
define void @store_i32_to_numeric_address() {
  %s = inttoptr i32 42 to ptr
  store i32 0, ptr %s
  ret void
}

; CHECK-LABEL: store_i32_to_global_address:
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.const $push1=, 0{{$}}
; CHECK: i32.store gv($pop0), $pop1{{$}}
define void @store_i32_to_global_address() {
  store i32 0, ptr @gv
  ret void
}

;===----------------------------------------------------------------------------
; Stores: 64-bit
;===----------------------------------------------------------------------------

; Basic store.

; CHECK-LABEL: store_i64_with_folded_offset:
; CHECK: i64.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  store i64 0, ptr %s
  ret void
}

; With an nuw add, we can fold an offset.

; CHECK-LABEL: store_i64_with_folded_gep_offset:
; CHECK: i64.store 24($0), $pop0{{$}}
define void @store_i64_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i64, ptr %p, i32 3
  store i64 0, ptr %s
  ret void
}

; With an inbounds gep, we can fold an offset.

; CHECK-LABEL: store_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_negative_offset(ptr %p) {
  %s = getelementptr inbounds i64, ptr %p, i32 -3
  store i64 0, ptr %s
  ret void
}

; We can't fold a negative offset though, even with an inbounds gep.

; CHECK-LABEL: store_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  store i64 0, ptr %s
  ret void
}

; Without nuw, and even with nsw, we can't fold an offset.

; CHECK-LABEL: store_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_offset(ptr %p) {
  %s = getelementptr i64, ptr %p, i32 3
  store i64 0, ptr %s
  ret void
}

; Without inbounds, we can't fold a gep offset.

; CHECK-LABEL: store_i32_with_folded_or_offset:
; CHECK: i32.store8 2($pop{{[0-9]+}}), $pop{{[0-9]+}}{{$}}
define void @store_i32_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to ptr
  %arrayidx = getelementptr inbounds i8, ptr %t0, i32 2
  store i8 0, ptr %arrayidx, align 1
  ret void
}

;===----------------------------------------------------------------------------
; Sign-extending loads
;===----------------------------------------------------------------------------

; Fold an offset into a sign-extending load.

; CHECK-LABEL: load_i8_i32_s_with_folded_offset:
; CHECK: i32.load8_s $push0=, 24($0){{$}}
define i32 @load_i8_i32_s_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i8, ptr %s
  %u = sext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i32_i64_s_with_folded_offset:
; CHECK: i64.load32_s $push0=, 24($0){{$}}
define i64 @load_i32_i64_s_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i32, ptr %s
  %u = sext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a sign-extending load.

; CHECK-LABEL: load_i8_i32_s_with_folded_gep_offset:
; CHECK: i32.load8_s $push0=, 24($0){{$}}
define i32 @load_i8_i32_s_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i8, ptr %p, i32 24
  %t = load i8, ptr %s
  %u = sext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i32_s_with_folded_gep_offset:
; CHECK: i32.load16_s $push0=, 48($0){{$}}
define i32 @load_i16_i32_s_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i16, ptr %p, i32 24
  %t = load i16, ptr %s
  %u = sext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i64_s_with_folded_gep_offset:
; CHECK: i64.load16_s $push0=, 48($0){{$}}
define i64 @load_i16_i64_s_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i16, ptr %p, i32 24
  %t = load i16, ptr %s
  %u = sext i16 %t to i64
  ret i64 %u
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: load_i8_i32_s_with_folded_or_offset:
; CHECK: i32.load8_s $push{{[0-9]+}}=, 2($pop{{[0-9]+}}){{$}}
define i32 @load_i8_i32_s_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to ptr
  %arrayidx = getelementptr inbounds i8, ptr %t0, i32 2
  %t1 = load i8, ptr %arrayidx
  %conv = sext i8 %t1 to i32
  ret i32 %conv
}

; CHECK-LABEL: load_i8_i64_s_with_folded_or_offset:
; CHECK: i64.load8_s $push{{[0-9]+}}=, 2($pop{{[0-9]+}}){{$}}
define i64 @load_i8_i64_s_with_folded_or_offset(i32 %x) {
  %and = and i32 %x, -4
  %t0 = inttoptr i32 %and to ptr
  %arrayidx = getelementptr inbounds i8, ptr %t0, i32 2
  %t1 = load i8, ptr %arrayidx
  %conv = sext i8 %t1 to i64
  ret i64 %conv
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i16_i32_s_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load16_s  $push1=, 42($pop0){{$}}
define i32 @load_i16_i32_s_from_numeric_address() {
  %s = inttoptr i32 42 to ptr
  %t = load i16, ptr %s
  %u = sext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i8_i32_s_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load8_s  $push1=, gv8($pop0){{$}}
@gv8 = global i8 0
define i32 @load_i8_i32_s_from_global_address() {
  %t = load i8, ptr @gv8
  %u = sext i8 %t to i32
  ret i32 %u
}

;===----------------------------------------------------------------------------
; Zero-extending loads
;===----------------------------------------------------------------------------

; Fold an offset into a zero-extending load.

; CHECK-LABEL: load_i8_i32_z_with_folded_offset:
; CHECK: i32.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_i32_z_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i8, ptr %s
  %u = zext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i32_i64_z_with_folded_offset:
; CHECK: i64.load32_u $push0=, 24($0){{$}}
define i64 @load_i32_i64_z_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = load i32, ptr %s
  %u = zext i32 %t to i64
  ret i64 %u
}

; Fold a gep offset into a zero-extending load.

; CHECK-LABEL: load_i8_i32_z_with_folded_gep_offset:
; CHECK: i32.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_i32_z_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i8, ptr %p, i32 24
  %t = load i8, ptr %s
  %u = zext i8 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i32_z_with_folded_gep_offset:
; CHECK: i32.load16_u $push0=, 48($0){{$}}
define i32 @load_i16_i32_z_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i16, ptr %p, i32 24
  %t = load i16, ptr %s
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i16_i64_z_with_folded_gep_offset:
; CHECK: i64.load16_u $push0=, 48($0){{$}}
define i64 @load_i16_i64_z_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i16, ptr %p, i64 24
  %t = load i16, ptr %s
  %u = zext i16 %t to i64
  ret i64 %u
}

; When loading from a fixed address, materialize a zero.

; CHECK-LABEL: load_i16_i32_z_from_numeric_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load16_u  $push1=, 42($pop0){{$}}
define i32 @load_i16_i32_z_from_numeric_address() {
  %s = inttoptr i32 42 to ptr
  %t = load i16, ptr %s
  %u = zext i16 %t to i32
  ret i32 %u
}

; CHECK-LABEL: load_i8_i32_z_from_global_address
; CHECK: i32.const $push0=, 0{{$}}
; CHECK: i32.load8_u  $push1=, gv8($pop0){{$}}
define i32 @load_i8_i32_z_from_global_address() {
  %t = load i8, ptr @gv8
  %u = zext i8 %t to i32
  ret i32 %u
}

; i8 return value should test anyext loads
; CHECK-LABEL: load_i8_i32_retvalue:
; CHECK: i32.load8_u $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i8 @load_i8_i32_retvalue(ptr %p) {
  %v = load i8, ptr %p
  ret i8 %v
}

;===----------------------------------------------------------------------------
; Truncating stores
;===----------------------------------------------------------------------------

; Fold an offset into a truncating store.

; CHECK-LABEL: store_i8_i32_with_folded_offset:
; CHECK: i32.store8 24($0), $1{{$}}
define void @store_i8_i32_with_folded_offset(ptr %p, i32 %v) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = trunc i32 %v to i8
  store i8 %t, ptr %s
  ret void
}

; CHECK-LABEL: store_i32_i64_with_folded_offset:
; CHECK: i64.store32 24($0), $1{{$}}
define void @store_i32_i64_with_folded_offset(ptr %p, i64 %v) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = trunc i64 %v to i32
  store i32 %t, ptr %s
  ret void
}

; Fold a gep offset into a truncating store.

; CHECK-LABEL: store_i8_i32_with_folded_gep_offset:
; CHECK: i32.store8 24($0), $1{{$}}
define void @store_i8_i32_with_folded_gep_offset(ptr %p, i32 %v) {
  %s = getelementptr inbounds i8, ptr %p, i32 24
  %t = trunc i32 %v to i8
  store i8 %t, ptr %s
  ret void
}

; CHECK-LABEL: store_i16_i32_with_folded_gep_offset:
; CHECK: i32.store16 48($0), $1{{$}}
define void @store_i16_i32_with_folded_gep_offset(ptr %p, i32 %v) {
  %s = getelementptr inbounds i16, ptr %p, i32 24
  %t = trunc i32 %v to i16
  store i16 %t, ptr %s
  ret void
}

; CHECK-LABEL: store_i16_i64_with_folded_gep_offset:
; CHECK: i64.store16 48($0), $1{{$}}
define void @store_i16_i64_with_folded_gep_offset(ptr %p, i64 %v) {
  %s = getelementptr inbounds i16, ptr %p, i64 24
  %t = trunc i64 %v to i16
  store i16 %t, ptr %s
  ret void
}

; 'add' in this code becomes 'or' after DAG optimization. Treat an 'or' node as
; an 'add' if the or'ed bits are known to be zero.

; CHECK-LABEL: store_i8_i32_with_folded_or_offset:
; CHECK: i32.store8 2($pop{{[0-9]+}}), $1{{$}}
define void @store_i8_i32_with_folded_or_offset(i32 %x, i32 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to ptr
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 2
  %t = trunc i32 %v to i8
  store i8 %t, ptr %arrayidx
  ret void
}

; CHECK-LABEL: store_i8_i64_with_folded_or_offset:
; CHECK: i64.store8 2($pop{{[0-9]+}}), $1{{$}}
define void @store_i8_i64_with_folded_or_offset(i32 %x, i64 %v) {
  %and = and i32 %x, -4
  %p = inttoptr i32 %and to ptr
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 2
  %t = trunc i64 %v to i8
  store i8 %t, ptr %arrayidx
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
define void @aggregate_load_store(ptr %p, ptr %q) {
  ; volatile so that things stay in order for the tests above
  %t = load volatile {i32,i32,i32,i32}, ptr %p
  store volatile {i32,i32,i32,i32} %t, ptr %q
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

;===----------------------------------------------------------------------------
; Loads: Half Precision
;===----------------------------------------------------------------------------

; Fold an offset into a zero-extending load.

; CHECK-LABEL: load_f16_f32_with_folded_offset:
; CHECK: f32.load_f16 $push0=, 24($0){{$}}
define float @load_f16_f32_with_folded_offset(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to ptr
  %t = call float @llvm.wasm.loadf16.f32(ptr %s)
  ret float %t
}

; Fold a gep offset into a zero-extending load.

; CHECK-LABEL: load_f16_f32_with_folded_gep_offset:
; CHECK: f32.load_f16 $push0=, 24($0){{$}}
define float @load_f16_f32_with_folded_gep_offset(ptr %p) {
  %s = getelementptr inbounds i8, ptr %p, i32 24
  %t = call float @llvm.wasm.loadf16.f32(ptr %s)
  ret float %t
}

;===----------------------------------------------------------------------------
; Stores: Half Precision
;===----------------------------------------------------------------------------

; Basic store.

; CHECK-LABEL: store_f16_f32_no_offset:
; CHECK-NEXT: .functype store_f16_f32_no_offset (i32, f32) -> (){{$}}
; CHECK-NEXT: f32.store_f16 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_f16_f32_no_offset(ptr %p, float %v) {
  call void @llvm.wasm.storef16.f32(float %v, ptr %p)
  ret void
}

; Storing to a fixed address.

; CHECK-LABEL: store_f16_f32_to_numeric_address:
; CHECK:      i32.const $push1=, 0{{$}}
; CHECK-NEXT: f32.const $push0=, 0x0p0{{$}}
; CHECK-NEXT: f32.store_f16 42($pop1), $pop0{{$}}
define void @store_f16_f32_to_numeric_address() {
  %s = inttoptr i32 42 to ptr
  call void @llvm.wasm.storef16.f32(float 0.0, ptr %s)
  ret void
}
