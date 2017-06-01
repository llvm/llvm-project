; RUN: opt < %s -loop-reduce -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

%struct.L = type { i8, i8* }

declare i32 @__CxxFrameHandler3(...)

@GV1 = external global %struct.L*
@GV2 = external global %struct.L

define void @b_copy_ctor() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %0 = load %struct.L*, %struct.L** @GV1, align 8
  br label %for.cond

for.cond:                                         ; preds = %call.i.noexc, %entry
  %d.0 = phi %struct.L* [ %0, %entry ], [ %incdec.ptr, %call.i.noexc ]
  invoke void @a_copy_ctor()
          to label %call.i.noexc unwind label %catch.dispatch

call.i.noexc:                                     ; preds = %for.cond
  %incdec.ptr = getelementptr inbounds %struct.L, %struct.L* %d.0, i64 1
  br label %for.cond

catch.dispatch:                                   ; preds = %for.cond
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* null, i32 64, i8* null]
  %cmp16 = icmp eq %struct.L* %0, %d.0
  br i1 %cmp16, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %catch
  %cmp = icmp eq %struct.L* @GV2, %d.0
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %catch
  catchret from %2 to label %try.cont

try.cont:                                         ; preds = %for.end
  ret void
}

; CHECK-LABEL: define void @b_copy_ctor(
; CHECK:       catchpad
; CHECK-NEXT:  icmp eq %struct.L
; CHECK-NEXT:  getelementptr {{.*}} i64 sub (i64 0, i64 ptrtoint (%struct.L* @GV2 to i64))

declare void @a_copy_ctor()
