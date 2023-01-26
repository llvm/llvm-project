; RUN: opt -S -passes=hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

declare void @sideeffect(i64)

declare ptr @realloc(ptr %ptr, i64 %size)

declare void @free(ptr %ptr)

declare void @sink() cold

; CHECK-LABEL: define {{.*}}@realloc2(
; CHECK: call {{.*}}@sideeffect(
; CHECK: call {{.*}}@realloc(
; CHECK-LABEL: codeRepl:
; CHECK: call {{.*}}@realloc2.cold.1(i64 %size, ptr %ptr, ptr %retval.0.ce.loc)
; CHECK-LABEL: cleanup:
; CHECK-NEXT: phi ptr [ null, %if.then ], [ %call, %if.end ], [ %retval.0.ce.reload, %codeRepl ]
define ptr @realloc2(ptr %ptr, i64 %size) {
entry:
  %0 = add i64 %size, -1
  %1 = icmp ugt i64 %0, 184549375
  br i1 %1, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sideeffect(i64 %size)
  br label %cleanup

if.end:                                           ; preds = %entry
  %call = call ptr @realloc(ptr %ptr, i64 %size)
  %tobool1 = icmp eq ptr %call, null
  br i1 %tobool1, label %if.then2, label %cleanup

if.then2:                                         ; preds = %if.end
  call void @sideeffect(i64 %size)
  call void @sink()
  %tobool3 = icmp eq ptr %ptr, null
  br i1 %tobool3, label %cleanup, label %if.then4

if.then4:                                         ; preds = %if.then2
  call void @free(ptr %ptr)
  br label %cleanup

cleanup:                                          ; preds = %if.end, %if.then4, %if.then2, %if.then
  %retval.0 = phi ptr [ null, %if.then ], [ null, %if.then2 ], [ null, %if.then4 ], [ %call, %if.end ]
  ret ptr %retval.0
}

; CHECK-LABEL: define {{.*}}@realloc2.cold.1(
; CHECK: call {{.*}}@sideeffect
; CHECK: call {{.*}}@sink
; CHECK: call {{.*}}@free
