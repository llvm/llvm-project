; RUN: opt -passes=loop-unroll -S < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

declare void @fn1(ptr)

declare i1 @fn2(ptr, ptr)

define void @fn4() personality ptr @__CxxFrameHandler3 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %i.05 = phi i8 [ 0, %entry ], [ %inc, %for.inc ]
  store i8 undef, ptr undef, align 4
  invoke void @fn1(ptr undef)
          to label %call.i.noexc unwind label %ehcleanup

call.i.noexc:                                     ; preds = %for.body
  %call1.i2 = invoke i1 @fn2(ptr undef, ptr undef)
          to label %call1.i.noexc unwind label %ehcleanup

call1.i.noexc:                                    ; preds = %call.i.noexc
  br i1 undef, label %if.then.i, label %if.end4.i

if.then.i:                                        ; preds = %call1.i.noexc
  %tmp1 = load i8, ptr undef, align 4
  %tobool.i = icmp eq i8 undef, undef
  br i1 undef, label %if.end4.i, label %if.then2.i

if.then2.i:                                       ; preds = %if.then.i
  %call3.i3 = invoke i1 @fn2(ptr undef, ptr null)
          to label %call3.i.noexc unwind label %ehcleanup

call3.i.noexc:                                    ; preds = %if.then2.i
  br label %if.end4.i

if.end4.i:                                        ; preds = %call3.i.noexc, %if.then.i, %call1.i.noexc
  %tmp2 = load i8, ptr undef, align 4
  br label %if.then6.i

if.then6.i:                                       ; preds = %if.end4.i
  %call7.i4 = invoke i1 @fn2(ptr undef, ptr null)
          to label %call7.i.noexc unwind label %ehcleanup

call7.i.noexc:                                    ; preds = %if.then6.i
  br label %fn3

fn3:                                              ; preds = %call7.i.noexc
  %tmp3 = load i8, ptr undef, align 4
  %inc.i = add nsw i8 undef, undef
  store i8 undef, ptr undef, align 4
  br label %for.inc

for.inc:                                          ; preds = %fn3
  %inc = add nsw i8 %i.05, 1
  %cmp = icmp slt i8 %inc, 6
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  invoke void @throw()
          to label %unreachable unwind label %ehcleanup

ehcleanup:                                        ; preds = %for.end, %if.then6.i, %if.then2.i, %call.i.noexc, %for.body
  %cp = cleanuppad within none []
  cleanupret from %cp unwind to caller

; CHECK: cleanuppad
; CHECK-NOT: cleanuppad

unreachable:                                      ; preds = %for.end
  unreachable
}

declare i32 @__CxxFrameHandler3(...)

declare void @throw()
