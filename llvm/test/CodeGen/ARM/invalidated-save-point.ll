; RUN: llc -mtriple thumbv7 -stop-after=if-converter < %s 2>&1 | FileCheck %s

; Make sure the save point and restore point are dropped from MFI at
; this point. Notably, if it isn't is will be invalid and reference a
; deleted block (%bb.-1.if.end)

; CHECK: savePoint: []
; CHECK: restorePoint: []

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7"

define i32 @f(i32 %n) {
entry:
  %cmp = icmp ult i32 %n, 4
  br i1 %cmp, label %return, label %if.end

if.end:
  tail call void @g(i32 %n)
  br label %return

return:
  %retval.0 = phi i32 [ 0, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

declare void @g(i32)
