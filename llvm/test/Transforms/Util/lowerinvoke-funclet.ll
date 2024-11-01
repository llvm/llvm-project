; RUN: opt -passes=lowerinvoke -S < %s | FileCheck %s

; Test if invoke instructions that have a funclet operand bundle can be lowered.

%struct.Cleanup = type { i8 }

define void @lowerinvoke_funclet() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: @lowerinvoke_funclet
entry:
  %c = alloca %struct.Cleanup, align 1
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  invoke void @bar(i32 3) [ "funclet"(token %1), "test"(i32 0) ]
          to label %invoke.cont1 unwind label %ehcleanup
; CHECK:  call void @bar(i32 3) [ "funclet"(token %1), "test"(i32 0) ]

invoke.cont1:                                     ; preds = %catch
  call void @"??1Cleanup@@QEAA@XZ"(ptr %c) #3 [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont1
  ret void

ehcleanup:                                        ; preds = %catch
  %2 = cleanuppad within %1 []
  call void @"??1Cleanup@@QEAA@XZ"(ptr %c) #3 [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

declare void @foo()
declare void @bar(i32)
declare i32 @__CxxFrameHandler3(...)
declare void @"??1Cleanup@@QEAA@XZ"(ptr) unnamed_addr
