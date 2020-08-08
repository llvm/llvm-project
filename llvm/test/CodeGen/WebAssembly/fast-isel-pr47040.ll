; RUN: llc < %s -fast-isel -fast-isel-abort=1 -verify-machineinstrs

; Regression test for PR47040, in which an assertion was improperly
; triggered during FastISel's address computation. The issue was that
; an `Address` set to be relative to FrameIndex zero was incorrectly
; considered to have an unset base. When the left hand side of an add
; set the Address to have a FrameIndex base of 0, the right side would
; not detect that the Address base had already been set and could try
; to set the Address to be relative to a register instead, triggering
; an assertion.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define i32 @foo() {
  %stack_addr = alloca i32
  %stack_i = ptrtoint i32* %stack_addr to i32
  %added = add i32 %stack_i, undef
  %added_addr = inttoptr i32 %added to i32*
  %ret = load i32, i32* %added_addr
  ret i32 %ret
}
