; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s
; Hot cold splitting should not outline:
; 1. Basic blocks with token type instructions
; 2. Functions with scoped EH personality

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.0"

; CHECK-LABEL: define {{.*}}@with_funclet
; CHECK-NOT: with_funclet.cold
define void @with_funclet() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @fYAXXZ()
          to label %normal unwind label %exception

normal:                                           ; preds = %entry
  ret void

exception:                                        ; preds = %entry
  %0 = cleanuppad within none []
  call void @terminateYAXXZ() [ "funclet"(token %0) ]
  br label %continueexception

continueexception:                                ; preds = %exception
  ret void
}

; CHECK-LABEL: define {{.*}}@with_personality
; CHECK-NOT: with_personality.cold
define void @with_personality(i32 %cond) personality ptr @__CxxFrameHandler3 {
entry:
  %cond.addr = alloca i32
  store i32 %cond, ptr %cond.addr
  %0 = load i32, ptr %cond.addr
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end2

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %cond.addr
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

declare i32 @__CxxFrameHandler3(...)

declare void @fYAXXZ()

declare void @bar() #0

declare void @terminateYAXXZ()

declare void @sideeffect(i32)

declare void @sink(...) #0

attributes #0 = { cold }
