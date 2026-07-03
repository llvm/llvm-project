; RUN: opt -S -verify-memoryssa -passes=loop-sink < %s | FileCheck %s
; RUN: opt -S -verify-memoryssa -aa-pipeline=basic-aa -passes=loop-sink < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.13.26128"

%struct.FontInfoData = type { ptr }
%struct.S = type { i8 }

; CHECK: @pr38462
; Make sure not to assert by trying to sink into catch.dispatch.

define void @pr38462(ptr %this) personality ptr @__C_specific_handler !prof !1 {
entry:
  %s = alloca %struct.S
  %call6 = call i32 @f()
  %tobool7 = icmp eq i32 %call6, 0
  br i1 %tobool7, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %call2 = invoke i32 @f() to label %__try.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %__except] unwind to caller

__except:
  %1 = catchpad within %0 [ptr null]
  catchret from %1 to label %__except3

__except3:
  call void @llvm.lifetime.start.p0(ptr nonnull %s)
  %call.i = call zeroext i1 @g(ptr nonnull %s)
  br i1 %call.i, label %if.then.i, label %exit

if.then.i:
  %call2.i = call i32 @f()
  br label %exit

exit:
  call void @llvm.lifetime.end.p0(ptr nonnull %s)
  br label %__try.cont

__try.cont:
  %call = call i32 @f()
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %for.body, label %for.cond.cleanup.loopexit
}

declare i32 @__C_specific_handler(...)
declare i32 @f()
declare zeroext i1 @g(ptr)
declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)

!1 = !{!"function_entry_count", i64 1}

