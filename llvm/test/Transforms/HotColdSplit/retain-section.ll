; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Retain section after cold code is split off.

; CHECK-LABEL: @fun
; CHECK: call void @fun.cold.1{{.*}}
define void @fun() section ".text.cold" {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  call void @sink()
  ret void
}

define void @foo() cold {
  ret void
}

declare void @sink() cold

; CHECK: define {{.*}} @fun.cold.1{{.*}} section ".text.cold"
