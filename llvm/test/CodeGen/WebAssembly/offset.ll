; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test constant load and store address offsets.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

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

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_folded_offset:
; CHECK: i64.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  %t = load i64, i64* %s
  ret i64 %t
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_folded_gep_offset:
; CHECK: i64.load  $push0=, 24($0){{$}}
define i64 @load_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  %t = load i64, i64* %s
  ret i64 %t
}

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  %t = load i64, i64* %s
  ret i64 %t
}

; Same as above but with i64.

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

; Same as above but with i64.

; CHECK-LABEL: load_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.load  $push2=, 0($pop1){{$}}
define i64 @load_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  %t = load i64, i64* %s
  ret i64 %t
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_folded_offset:
; CHECK: i32.store $discard=, 24($0), $pop0{{$}}
define void @store_i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store i32 0, i32* %s
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_folded_gep_offset:
; CHECK: i32.store $discard=, 24($0), $pop0{{$}}
define void @store_i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 6
  store i32 0, i32* %s
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store $discard=, 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -6
  store i32 0, i32* %s
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store $discard=, 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i32*
  store i32 0, i32* %s
  ret void
}

; Same as above but with store.

; CHECK-LABEL: store_i32_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i32.store $discard=, 0($pop1), $pop2{{$}}
define void @store_i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 6
  store i32 0, i32* %s
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_folded_offset:
; CHECK: i64.store $discard=, 24($0), $pop0{{$}}
define void @store_i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store i64 0, i64* %s
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_folded_gep_offset:
; CHECK: i64.store $discard=, 24($0), $pop0{{$}}
define void @store_i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 3
  store i64 0, i64* %s
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_unfolded_gep_negative_offset:
; CHECK: i32.const $push0=, -24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store $discard=, 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -3
  store i64 0, i64* %s
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_unfolded_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store $discard=, 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 24
  %s = inttoptr i32 %r to i64*
  store i64 0, i64* %s
  ret void
}

; Same as above but with store with i64.

; CHECK-LABEL: store_i64_with_unfolded_gep_offset:
; CHECK: i32.const $push0=, 24{{$}}
; CHECK: i32.add   $push1=, $0, $pop0{{$}}
; CHECK: i64.store $discard=, 0($pop1), $pop2{{$}}
define void @store_i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 3
  store i64 0, i64* %s
  ret void
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

; CHECK-LABEL: store_i32_to_numeric_address:
; CHECK: i32.const $0=, 0{{$}}
; CHECK: i32.store $discard=, 42($0), $0{{$}}
define void @store_i32_to_numeric_address() {
  %s = inttoptr i32 42 to i32*
  store i32 0, i32* %s
  ret void
}

; CHECK-LABEL: store_i32_to_global_address:
; CHECK: i32.const $0=, 0{{$}}
; CHECK: i32.store $discard=, gv($0), $0{{$}}
define void @store_i32_to_global_address() {
  store i32 0, i32* @gv
  ret void
}

; Fold an offset into a sign-extending load.

; CHECK-LABEL: load_i8_s_with_folded_offset:
; CHECK: i32.load8_s $push0=, 24($0){{$}}
define i32 @load_i8_s_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load i8, i8* %s
  %u = sext i8 %t to i32
  ret i32 %u
}

; Fold a gep offset into a sign-extending load.

; CHECK-LABEL: load_i8_s_with_folded_gep_offset:
; CHECK: i32.load8_s $push0=, 24($0){{$}}
define i32 @load_i8_s_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load i8, i8* %s
  %u = sext i8 %t to i32
  ret i32 %u
}

; Fold an offset into a zero-extending load.

; CHECK-LABEL: load_i8_u_with_folded_offset:
; CHECK: i32.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_u_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  %t = load i8, i8* %s
  %u = zext i8 %t to i32
  ret i32 %u
}

; Fold a gep offset into a zero-extending load.

; CHECK-LABEL: load_i8_u_with_folded_gep_offset:
; CHECK: i32.load8_u $push0=, 24($0){{$}}
define i32 @load_i8_u_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  %t = load i8, i8* %s
  %u = zext i8 %t to i32
  ret i32 %u
}

; Fold an offset into a truncating store.

; CHECK-LABEL: store_i8_with_folded_offset:
; CHECK: i32.store8 $discard=, 24($0), $pop0{{$}}
define void @store_i8_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 24
  %s = inttoptr i32 %r to i8*
  store i8 0, i8* %s
  ret void
}

; Fold a gep offset into a truncating store.

; CHECK-LABEL: store_i8_with_folded_gep_offset:
; CHECK: i32.store8 $discard=, 24($0), $pop0{{$}}
define void @store_i8_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 24
  store i8 0, i8* %s
  ret void
}

; Fold the offsets when lowering aggregate loads and stores.

; CHECK-LABEL: aggregate_load_store:
; CHECK: i32.load  $2=, 0($0){{$}}
; CHECK: i32.load  $3=, 4($0){{$}}
; CHECK: i32.load  $4=, 8($0){{$}}
; CHECK: i32.load  $push0=, 12($0){{$}}
; CHECK: i32.store $discard=, 12($1), $pop0{{$}}
; CHECK: i32.store $discard=, 8($1), $4{{$}}
; CHECK: i32.store $discard=, 4($1), $3{{$}}
; CHECK: i32.store $discard=, 0($1), $2{{$}}
define void @aggregate_load_store({i32,i32,i32,i32}* %p, {i32,i32,i32,i32}* %q) {
  ; volatile so that things stay in order for the tests above
  %t = load volatile {i32,i32,i32,i32}, {i32, i32,i32,i32}* %p
  store volatile {i32,i32,i32,i32} %t, {i32, i32,i32,i32}* %q
  ret void
}

; Fold the offsets when lowering aggregate return values.

; CHECK-LABEL: aggregate_return:
; CHECK: i32.const   $push0=, 0{{$}}
; CHECK: i32.store   $push1=, 12($0), $pop0{{$}}
; CHECK: i32.store   $push2=, 8($0), $pop1{{$}}
; CHECK: i32.store   $push3=, 4($0), $pop2{{$}}
; CHECK: i32.store   $discard=, 0($0), $pop3{{$}}
define {i32,i32,i32,i32} @aggregate_return() {
  ret {i32,i32,i32,i32} zeroinitializer
}
