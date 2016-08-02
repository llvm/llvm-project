; RUN: opt -prune-eh -inline -jump-threading -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@d = external global i32*, align 8

define void @fn3(i1 %B) {
entry:
  br i1 %B, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @fn2()
  ret void

if.end:                                           ; preds = %entry
  call void @fn2()
  ret void
}

define internal void @fn2() unnamed_addr {
entry:
  call void @fn1()
  call void @fn1()
  call void @fn1()
  unreachable
}

; CHECK-LABEL: define internal void @fn2(
; CHECK:   %[[LOAD:.*]] = load i32*, i32** @d, align 8
; CHECK:   %tobool1.i = icmp eq i32* %[[LOAD]], null

define internal void @fn1() unnamed_addr {
entry:
  br label %for.body

for.body:                                         ; preds = %entry
  %0 = load i32*, i32** @d, align 8
  %tobool1 = icmp eq i32* %0, null
  br i1 %tobool1, label %cond.false, label %cond.end

cond.false:                                       ; preds = %for.body
  call void @__assert_fail(i8* null)
  unreachable

cond.end:                                         ; preds = %for.body
  %1 = load i32*, i32** @d, align 8
  %cmp = icmp eq i32* %1, null
  br i1 %cmp, label %cond.end4, label %cond.false3

cond.false3:                                      ; preds = %cond.end
  call void @__assert_fail(i8* null)
  unreachable

cond.end4:                                        ; preds = %cond.end
  call void @__assert_fail(i8* null)
  unreachable

for.end:                                          ; No predecessors!
  ret void
}

declare void @__assert_fail(i8*)

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #0

attributes #0 = { noreturn nounwind }
