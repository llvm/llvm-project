; RUN: opt -mtriple=x86_64-windows-msvc -passes=objc-arc -S < %s | FileCheck %s

; Check that funclet tokens are preserved
;
; CHECK-LABEL:  catch:
; CHECK:          %1 = catchpad within %0
; CHECK:          %2 = tail call ptr @llvm.objc.retain(ptr %exn) #0 [ "funclet"(token %1) ]
; CHECK:          call void @llvm.objc.release(ptr %exn) #0 [ "funclet"(token %1) ]
; CHECK:          catchret from %1 to label %eh.cont

define void @try_catch_with_objc_intrinsic() personality ptr @__CxxFrameHandler3 {
entry:
  %exn.slot = alloca ptr, align 8
  invoke void @may_throw(ptr null) to label %eh.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

eh.cont:                                          ; preds = %catch, %entry
  ret void

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 0, ptr %exn.slot]
  br label %if.then

if.then:                                          ; preds = %catch
  %exn = load ptr, ptr null, align 8
  %2 = call ptr @llvm.objc.retain(ptr %exn) [ "funclet"(token %1) ]
  call void @may_throw(ptr %exn)
  call void @llvm.objc.release(ptr %exn) [ "funclet"(token %1) ]
  catchret from %1 to label %eh.cont
}

declare void @may_throw(ptr)
declare i32 @__CxxFrameHandler3(...)

declare ptr @llvm.objc.retain(ptr) #0
declare void @llvm.objc.release(ptr) #0

attributes #0 = { nounwind }
