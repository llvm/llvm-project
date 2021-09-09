; RUN: llc < %s -verify-machineinstrs | FileCheck %s

; Wasm does not currently support function addresses with offsets, so we
; shouldn't try to create a folded SDNode like (function + offset). This is a
; regression test for the folding bug and this should not crash in MCInstLower.

target triple = "wasm32-unknown-unknown"

; 'hidden' here should be present to reproduce the bug
declare hidden void @ham(i8*)

define void @bar(i8* %ptr) {
bb1:
  br i1 undef, label %bb3, label %bb2

bb2:
  ; While lowering this switch, isel creates (@ham + 1) expression as a course
  ; of range optimization for switch, and tries to fold the expression, but
  ; wasm does not support with function addresses with offsets. This folding
  ; should be disabled.
  ; CHECK:      i32.const  ham
  ; CHECK-NEXT: i32.const  1
  ; CHECK-NEXT: i32.add
  switch i32 ptrtoint (void (i8*)* @ham to i32), label %bb4 [
    i32 -1, label %bb3
    i32 0, label %bb3
  ]

bb3:
  unreachable

bb4:
  %tmp = load i8, i8* %ptr
  unreachable
}
