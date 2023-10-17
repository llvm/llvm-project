; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @report_missing() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw() to label %eh.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %catch] unwind to caller

catch:
  %1 = catchpad within %0 [ptr null, i32 0, ptr null]
  br label %catch.cont

catch.cont:
; CHECK: Missing funclet token on intrinsic call
  %2 = call ptr @llvm.objc.retain(ptr null)
  catchret from %1 to label %eh.cont

eh.cont:
  ret void
}

declare void @may_throw()
declare i32 @__CxxFrameHandler3(...)

declare ptr @llvm.objc.retain(ptr) #0

attributes #0 = { nounwind }
