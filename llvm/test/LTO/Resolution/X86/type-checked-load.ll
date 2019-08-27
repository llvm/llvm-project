; RUN: llvm-as -o %t %s
; RUN: llvm-lto2 run %t -O0 -r %t,foo,px -o %t2

; This just tests that we don't crash when compiling this test case.
; It means that the wholeprogramdevirt pass must have run and lowered
; the llvm.type.checked.load call.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define {i8*, i1} @foo(i8* %ptr) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %ptr, i32 16, metadata !"foo")
  ret {i8*, i1} %pair
}

declare {i8*, i1} @llvm.type.checked.load(i8* %ptr, i32 %offset, metadata %type)
