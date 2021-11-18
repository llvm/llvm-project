; RUN: opt -S -passes=globaldce < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@if = ifunc void (), void ()* ()* @fn

define internal void ()* @fn() {
entry:
  ret void ()* null
}

; CHECK-DAG: @if = ifunc void (), void ()* ()* @fn
; CHECK-DAG: define internal void ()* @fn(
