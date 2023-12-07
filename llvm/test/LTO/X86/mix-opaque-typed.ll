; RUN: llvm-as %S/Inputs/opaque-pointers.ll -o %t-opaque.bc
; RUN: llvm-lto2 run -o %t-lto.bc %S/Inputs/typed.bc %t-opaque.bc -save-temps \
; RUN:     -r %S/Inputs/typed.bc,call_foo,px -r %S/Inputs/typed.bc,foo,l \
; RUN:     -r %t-opaque.bc,foo,px
; RUN: opt -S -o - %t-lto.bc.0.4.opt.bc | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64 @foo(i64* %p);

define i64 @call_foo(i64* %p) {
  ; CHECK-LABEL: define i64 @call_foo(ptr nocapture readonly %p) local_unnamed_addr #0 {
  ; CHECK-NEXT: %t.i = load i64, ptr %p, align 8
  %t = call i64 @foo(i64* %p)
  ret i64 %t
}
