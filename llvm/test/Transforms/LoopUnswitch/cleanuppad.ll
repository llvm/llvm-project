; RUN: opt -S -loop-unswitch < %s | FileCheck %s
target triple = "x86_64-pc-win32"

define void @f(i32 %doit, i1 %x, i1 %y) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %tobool = icmp eq i32 %doit, 0
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  br i1 %x, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br i1 %tobool, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  br i1 %y, label %for.inc, label %delete.notnull

delete.notnull:                                   ; preds = %if.then
  invoke void @g()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %delete.notnull
  br label %for.inc

lpad:                                             ; preds = %delete.notnull
  %cp = cleanuppad within none []
  cleanupret from %cp unwind to caller

for.inc:                                          ; preds = %invoke.cont, %if.then, %for.body
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @g()

declare i32 @__CxxFrameHandler3(...)

; CHECK-LABEL: define void @f(
; CHECK: cleanuppad within none []
; CHECK-NOT: cleanuppad

attributes #0 = { ssp uwtable }
