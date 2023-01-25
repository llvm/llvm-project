; RUN: opt -S -passes=hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

; Source:
;
; extern void sideeffect(int);
; extern void __attribute__((cold)) sink();
; void foo(int cond) {
;   if (cond) { //< Start outlining here.
;     sink();
;     if (cond > 10)
;       goto exit1;
;     else
;       goto exit2;
;   }
; exit1:
;   sideeffect(1);
;   return;
; exit2:
;   sideeffect(2);
;   return;
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK: br i1 {{.*}}, label %exit1, label %codeRepl
; CHECK-LABEL: codeRepl:
; CHECK: [[targetBlock:%.*]] = call i1 @foo.cold.1(
; CHECK-NEXT: br i1 [[targetBlock]], label %exit1, label %[[return:.*]]
; CHECK-LABEL: exit1:
; CHECK: call {{.*}}@sideeffect(i32 1)
; CHECK: [[return]]:
; CHECK-NEXT: ret void
define void @foo(i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %exit1, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...) @sink()
  %cmp = icmp sgt i32 %cond, 10
  br i1 %cmp, label %exit1, label %exit2

exit1:                                            ; preds = %entry, %if.then
  call void @sideeffect(i32 1)
  br label %return

exit2:                                            ; preds = %if.then
  call void @sideeffect(i32 2)
  br label %return

return:                                           ; preds = %exit2, %exit1
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK: br
; CHECK: call {{.*}}@sink
; CHECK-NEXT: [[cmp:%.*]] = icmp
; CHECK-NEXT: br i1 [[cmp]], label %[[exit1Stub:.*]], label %exit2
; CHECK-LABEL: exit2:
; CHECK-NEXT: call {{.*}}@sideeffect(i32 2)
; CHECK-NEXT: br label %[[returnStub:.*]]
; CHECK: [[exit1Stub]]:
; CHECK-NEXT: ret i1 true
; CHECK: [[returnStub]]:
; CHECK-NEXT: ret i1 false

declare void @sink(...) cold

declare void @sideeffect(i32)
