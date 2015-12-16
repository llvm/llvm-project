; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test constant load and store address offsets.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
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
