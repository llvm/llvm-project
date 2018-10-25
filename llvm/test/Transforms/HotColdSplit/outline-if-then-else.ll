; RUN: opt -S -hotcoldsplit < %s | FileCheck %s

; Source:
;
; extern void sideeffect(int);
; extern void __attribute__((cold)) sink();
; void foo(int cond) {
;   if (cond) { //< Start outlining here.
;     if (cond > 10)
;       sideeffect(0);
;     else
;       sideeffect(1);
;     sink();
;   }
;   sideeffect(2);
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK: br i1 {{.*}}, label %codeRepl, label %if.end2
; CHECK-LABEL: codeRepl:
; CHECK-NEXT: call void @foo.cold.1
; CHECK-LABEL: if.end2:
; CHECK: call void @sideeffect(i32 2)
define void @foo(i32 %cond) {
entry:
  %cond.addr = alloca i32
  store i32 %cond, i32* %cond.addr
  %0 = load i32, i32* %cond.addr
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end2

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %cond.addr
  %cmp = icmp sgt i32 %1, 10
  br i1 %cmp, label %if.then1, label %if.else

if.then1:                                         ; preds = %if.then
  call void @sideeffect(i32 0)
  br label %if.end

if.else:                                          ; preds = %if.then
  call void @sideeffect(i32 1)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then1
  call void (...) @sink()
  ret void

if.end2:                                          ; preds = %entry
  call void @sideeffect(i32 2)
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1
; CHECK: call {{.*}}@sideeffect
; CHECK: call {{.*}}@sideeffect
; CHECK: call {{.*}}@sink

declare void @sideeffect(i32)

declare void @sink(...) cold
