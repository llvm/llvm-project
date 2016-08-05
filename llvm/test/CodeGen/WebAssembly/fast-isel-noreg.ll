; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -fast-isel -verify-machineinstrs | FileCheck %s

; Test that FastISel does not generate instructions with NoReg

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: i32.const $push0=, 0
define hidden i32 @a() #0 {
entry:
  ret i32 zext (i1 icmp eq (void (...)* inttoptr (i32 10 to void (...)*), void (...)* null) to i32)
}

; CHECK: i32.const $push0=, 1
; CHECK: br_if 0, $pop0
define hidden i32 @b() #0 {
entry:
  br i1 icmp eq (void (...)* inttoptr (i32 10 to void (...)*), void (...)* null), label %a, label %b
a:
  unreachable
b:
  ret i32 0
}

; CHECK: i32.const $push1=, 0
; CHECK: i32.const $push2=, 0
; CHECK: i32.store $drop=, 0($pop1), $pop2
define hidden i32 @c() #0 {
entry:
  store i32 zext (i1 icmp eq (void (...)* inttoptr (i32 10 to void (...)*), void (...)* null) to i32), i32* inttoptr (i32 0 to i32 *)
  ret i32 0
}

attributes #0 = { noinline optnone }
