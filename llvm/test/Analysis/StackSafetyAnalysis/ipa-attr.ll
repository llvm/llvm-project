; RUN: llvm-as %s -o %t0.bc
; RUN: llvm-as %S/Inputs/ipa.ll -o %t1.bc
; RUN: llvm-link -disable-lazy-loading %t0.bc %t1.bc -o %t.combined.bc
; RUN: opt -S -passes="stack-safety-annotator" %t.combined.bc -o - 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @Write1(i8* %p)
declare void @Write8(i8* %p)

; Basic out-of-bounds.
define void @f1() {
; CHECK-LABEL: define void @f1() {
; CHECK: alloca i32, align 4{{$}}
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @Write8(i8* %x1)
  ret void
}

; Basic in-bounds.
define void @f2() {
; CHECK-LABEL: define void @f2() {
; CHECK: alloca i32, align 4, !stack-safe ![[A:[0-9]+]]{{$}}
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @Write1(i8* %x1)
  ret void
}

; CHECK: ![[A]] = !{}
