; RUN: opt -S -loop-reduce < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

define void @fn3() personality ptr @__CxxFrameHandler3 {
entry:
  %call = invoke i32 @fn2()
          to label %for.cond.preheader unwind label %catch.dispatch2

for.cond.preheader:                               ; preds = %entry
  br label %for.cond

for.cond:                                         ; preds = %for.cond.preheader, %for.cond
  %b.0 = phi i32 [ %inc, %for.cond ], [ %call, %for.cond.preheader ]
  %inc = add nsw i32 %b.0, 1
  invoke void @fn1(i32 %inc)
          to label %for.cond unwind label %catch.dispatch

; CHECK:   %[[add:.*]] = add i32 %call, 1
; CHECK:   br label %for.cond

; CHECK: for.cond:                                         ; preds = %for.cond, %for.cond.preheader
; CHECK:   %[[lsr_iv:.*]] = phi i32 [ %lsr.iv.next, %for.cond ], [ %[[add]], %for.cond.preheader ]
; CHECK:   %[[lsr_iv_next:.*]] = add i32 %lsr.iv, 1
; CHECK:   invoke void @fn1(i32 %[[lsr_iv]])


catch.dispatch:                                   ; preds = %for.cond
  %0 = catchswitch within none [label %catch] unwind label %catch.dispatch2

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  invoke void @_CxxThrowException(ptr null, ptr null) #2 [ "funclet"(token %1) ]
          to label %unreachable unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %catch.dispatch, %catch, %entry
  %a.0 = phi i32 [ undef, %entry ], [ %call, %catch ], [ %call, %catch.dispatch ]
  %2 = catchswitch within none [label %catch3] unwind to caller

catch3:                                           ; preds = %catch.dispatch2
  %3 = catchpad within %2 [ptr null, i32 64, ptr null]
  call void @fn1(i32 %a.0) [ "funclet"(token %3) ]
  catchret from %3 to label %try.cont4

try.cont4:                                        ; preds = %catch3
  ret void

unreachable:                                      ; preds = %catch
  unreachable
}

declare i32 @fn2()

declare i32 @__CxxFrameHandler3(...)

declare void @fn1(i32)

declare void @_CxxThrowException(ptr, ptr)
