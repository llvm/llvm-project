; RUN: opt -S -passes='require<memoryssa>' %s | FileCheck %s
; REQUIRES: asserts
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @func()
; Function Attrs: noinline
define dso_local void @func() unnamed_addr #0 align 2 {
entry:
  %NoFinalize.addr = alloca i8, align 1
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  %call8 = call zeroext i1 @foo()
  br i1 %call8, label %if.then9, label %while.cond

if.then9:                                         ; preds = %entry
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  br label %while.cond

while.cond:                                       ; preds = %cleanup, %if.then9, %entry
  %call34 = call zeroext i1 @foo()
  call void @blah()
  br i1 %call34, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %call35 = call zeroext i1 @foo()
  br i1 %call35, label %if.end37, label %if.then36

if.then36:                                        ; preds = %while.body
  store i32 2, ptr undef, align 4
  br label %cleanup

if.end37:                                         ; preds = %while.body
  %call38 = call zeroext i1 @foo()
  br i1 %call38, label %if.end46, label %land.lhs.true

land.lhs.true:                                    ; preds = %if.end37
  call void @blah()
  %call41 = call zeroext i1 @foo()
  br i1 %call41, label %if.then42, label %if.end46

if.then42:                                        ; preds = %land.lhs.true
  call void @blah()
  br label %if.end46

if.end46:                                         ; preds = %if.then42, %land.lhs.true, %if.end37
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  br label %cleanup

cleanup:                                          ; preds = %if.end46, %if.then36
  call void @blah()
  br label %while.cond

while.end:                                        ; preds = %while.cond
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  %call93 = call zeroext i1 @foo()
  br i1 %call93, label %if.end120, label %if.then94

if.then94:                                        ; preds = %while.end
  store i32 0, ptr undef, align 4
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.then94
  br i1 undef, label %for.body, label %if.end120

for.body:                                         ; preds = %for.cond
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  call void @blah()
  br label %for.cond

if.end120:                                        ; preds = %for.cond, %while.end
  %val = load i8, ptr %NoFinalize.addr, align 1
  ret void
}

; Function Attrs: noinline
declare hidden void @blah() unnamed_addr #0 align 2

; Function Attrs: noinline
declare hidden i1 @foo() local_unnamed_addr #0 align 2

attributes #0 = { noinline }

