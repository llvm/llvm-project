; RUN: opt -S -lowertypetests -lowertypetests-summary-action=export -o - %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

; CHECK: @dipsy = external
@dipsy = external constant i8, !type !0

define void @tinkywinky() {
  store i8* @dipsy, i8** undef
  ret void
}

!0 = !{i64 16, !"teletubbies"}
