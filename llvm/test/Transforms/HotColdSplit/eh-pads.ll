; RUN: opt -S -hotcoldsplit < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK: landingpad
; CHECK: sideeffect(i32 2)
define void @foo(i32 %cond) personality i8 0 {
entry:
  invoke void @llvm.donothing() to label %normal unwind label %exception

exception:
  ; Note: EH pads are not candidates for region entry points.
  %cleanup = landingpad i8 cleanup
  br label %continue_exception

continue_exception:
  call void @sideeffect(i32 0)
  call void @sideeffect(i32 1)
  call void @sink()
  ret void

normal:
  call void @sideeffect(i32 2)
  ret void
}

; See llvm.org/PR39917. It's currently not safe to outline landingpad
; instructions.
;
; CHECK-LABEL: define {{.*}}@bar(
; CHECK: landingpad
define void @bar(i32 %cond) personality i8 0 {
entry:
  br i1 undef, label %exit, label %continue

exit:
  ret void

continue:
  invoke void @sink() to label %normal unwind label %exception

exception:
  ; Note: EH pads are not candidates for region entry points.
  %cleanup = landingpad i8 cleanup
  ret void

normal:
  call void @sideeffect(i32 0)
  call void @sideeffect(i32 1)
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK: sideeffect(i32 0)
; CHECK: sideeffect(i32 1)
; CHECK: sink

declare void @sideeffect(i32)

declare void @sink() cold

declare void @llvm.donothing() nounwind readnone
