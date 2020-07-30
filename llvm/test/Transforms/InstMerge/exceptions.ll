; RUN: opt -basic-aa -memdep -mldst-motion -S < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<memdep>',mldst-motion \
; RUN:   -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@r = common global i32 0, align 4
@s = common global i32 0, align 4

; CHECK-LABEL: define void @test1(
define void @test1(i1 %cmp, i32* noalias %p) {
entry:
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  call void @may_exit() nounwind
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 1
  %0 = load i32, i32* %arrayidx, align 4
  store i32 %0, i32* @r, align 4
  br label %if.end
; CHECK:       call void @may_exit()
; CHECK-NEXT:  %[[gep:.*]] = getelementptr inbounds i32, i32* %p, i64 1
; CHECK-NEXT:  %[[load:.*]] = load i32, i32* %[[gep]], align 4
; CHECK-NEXT:  store i32 %[[load]], i32* @r, align 4

if.else:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 1
  %1 = load i32, i32* %arrayidx1, align 4
  store i32 %1, i32* @s, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK-LABEL: define void @test2(
define void @test2(i1 %cmp, i32* noalias %p) {
entry:
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 1
  store i32 1, i32* %arrayidx, align 4
  call void @may_throw()
; CHECK:       %[[gep:.*]] = getelementptr inbounds i32, i32* %p, i64 1
; CHECK-NEXT:  store i32 1, i32* %[[gep]], align 4
; CHECK-NEXT:  call void @may_throw()
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 1
  store i32 2, i32* %arrayidx1, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare void @may_throw()
declare void @may_exit() nounwind
