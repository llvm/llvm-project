; RUN: opt -indvars -S < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

define i32 @fn2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %c.0 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
; CHECK: %[[WIDE:.*]] = phi i64
; CHECK: %[[NORM:.*]] = phi i32
; CHECK: invoke void @fn1(i64 %[[WIDE]])
  %idxprom = sext i32 %c.0 to i64
  invoke void @fn1(i64 %idxprom)
          to label %for.inc unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %for.cond
  %c.0.lcssa = phi i32 [ %c.0, %for.cond ]
; CHECK: %[[LCSSA:.*]] = phi i32 [ %[[NORM]],
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  catchret from %1 to label %exit

exit:
; CHECK: ret i32 %[[LCSSA]]
  ret i32 %c.0.lcssa

for.inc:                                          ; preds = %for.cond
  %inc = add nsw nuw i32 %c.0, 1
  br label %for.cond
}

declare void @fn1(i64 %idxprom)

declare i32 @__CxxFrameHandler3(...)
