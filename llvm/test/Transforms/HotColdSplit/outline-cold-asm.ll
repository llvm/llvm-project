; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@fun(
; CHECK: call {{.*}}@fun.cold.1(

; CHECK-LABEL: define {{.*}}@fun.cold.1(
; CHECK: asm ""

define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  call void asm "", ""()
  call void @sink()
  ret void
}

declare void @sink() cold
