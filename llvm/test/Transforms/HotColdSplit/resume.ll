; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Consider `resume` to be cold.

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK: call {{.*}}@sink(

declare void @sink() cold

define i32 @foo() personality i8 0 {
entry:
  br i1 undef, label %pre-resume-eh, label %normal

pre-resume-eh:
  call void @sink()
  br label %resume-eh

resume-eh:
  resume i32 undef

normal:
  ret i32 0
}
