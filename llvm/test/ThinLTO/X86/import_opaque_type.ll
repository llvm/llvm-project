; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/import_opaque_type.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; RUN: llvm-lto -thinlto-action=import %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s


target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; FIXME: It would be better to produce %a = type { i8 } here
; CHECK: %a = type opaque
%a = type opaque

@g = external global %a

define ptr @test() {
  ret ptr @g
}
