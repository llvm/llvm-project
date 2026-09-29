; RUN: opt -S -passes=hotcoldsplit -hotcoldsplit-max-params=1 < %s | FileCheck %s

target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc64-ibm-aix7.2.0.0"

define void @foo(i32 %cond) {
; CHECK-LABEL: define {{.*}}@foo(
; CHECK:       if.then:
; CHECK:       br i1 {{.*}}, label %if.then1, label %codeRepl
; CHECK-LABEL: codeRepl:
; CHECK-NEXT:  call void @foo.cold.1
;
entry:
  %cond.addr = alloca i32
  store i32 %cond, ptr %cond.addr
  %0 = load i32, ptr %cond.addr
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end2

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %cond.addr
  call void @sink(i32 %0)
  %cmp = icmp sgt i32 %1, 10
  br i1 %cmp, label %if.then1, label %if.else

if.then1:                                         ; preds = %if.then
  call void @sideeffect(i32 2)
  br label %if.end

if.else:                                          ; preds = %if.then
  call void @sink(i32 0)
  call void @sideeffect(i32 0)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then1
  br label %if.end2

if.end2:                                          ; preds = %entry
  call void @sideeffect(i32 1)
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1
; CHECK:       call {{.*}}@sink
; CHECK-NEXT:  call {{.*}}@sideeffect

declare void @sideeffect(i32)

declare void @sink(i32) cold
