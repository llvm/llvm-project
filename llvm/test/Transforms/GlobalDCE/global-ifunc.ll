; RUN: opt -S -passes=globaldce < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@if = ifunc void (), ptr @fn

define internal ptr @fn() {
entry:
  ret ptr null
}

; CHECK-DAG: @if = ifunc void (), ptr @fn
; CHECK-DAG: define internal ptr @fn(
