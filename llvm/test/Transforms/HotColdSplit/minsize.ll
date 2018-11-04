; RUN: opt -hotcoldsplit -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: @fun
; CHECK: call void @fun.cold.1
define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  call void @sink()
  call void @sink()
  call void @sink()
  ret void
}

declare void @sink() cold

; CHECK: define {{.*}} @fun.cold.1{{.*}}#[[outlined_func_attr:[0-9]+]]
; CHECK: attributes #[[outlined_func_attr]] = { {{.*}}minsize
