; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@fun
; CHECK: call {{.*}}@fun.cold.1(
define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ; This will be marked by the inverse DFS on sink-predecesors.
  br label %sink

sink:
  call void @sink()

  ; Do not allow the forward-DFS on sink-successors to mark the block again.
  br i1 undef, label %if.then, label %if.then.exit

if.then.exit:
  ret void

if.else:
  ret void
}

declare void @sink() cold
