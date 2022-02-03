; RUN: opt -mtriple=i686-unknown-windows-msvc -objc-arc-contract -S -o - %s | FileCheck %s

; Generated (and lightly modified and cleaned up) from the following source:
; id f();
; void g() {
;   try {
;     f();
;   } catch (...) {
;     f();
;   }
; }

; CHECK-LABEL: define void @"\01?g@@YAXXZ"()
; CHECK-LABEL: catch
; CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""() [ "funclet"(token %1) ]

; CHECK-LABEL: catch.1
; CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""() [ "funclet"(token %1) ]

; CHECK-LABEL: invoke.cont
; CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""(){{$}}

define void @"\01?g@@YAXXZ"() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %call = invoke i8* @"\01?f@@YAPAUobjc_object@@XZ"()
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  %call1 = call i8* @"\01?f@@YAPAUobjc_object@@XZ"() [ "funclet"(token %1) ]
  %2 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call1) [ "funclet"(token %1) ]
  call void @llvm.objc.release(i8* %2) [ "funclet"(token %1) ]
  br label %catch.1

catch.1:                                          ; preds = %catch
  %call2 = call i8* @"\01?f@@YAPAUobjc_object@@XZ"() [ "funclet"(token %1) ]
  %3 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call2) [ "funclet"(token %1) ]
  call void @llvm.objc.release(i8* %3) [ "funclet"(token %1) ]
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch.1
  ret void

invoke.cont:                                      ; preds = %entry
  %4 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call)
  call void @llvm.objc.release(i8* %4)
  ret void
}

; CHECK-LABEL: define dso_local void @"?test_attr_claimRV@@YAXXZ"()
; CHECK: %[[CALL4:.*]] = notail call i8* @"?noexcept_func@@YAPAUobjc_object@@XZ"() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

; CHECK: %[[V1:.*]] = cleanuppad
; CHECK: %[[CALL:.*]] = notail call i8* @"?noexcept_func@@YAPAUobjc_object@@XZ"() [ "funclet"(token %[[V1]]), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NEXT: cleanupret from %[[V1]] unwind to caller

define dso_local void @"?test_attr_claimRV@@YAXXZ"() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @"?foo@@YAXXZ"()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %call.i4 = tail call i8* @"?noexcept_func@@YAPAUobjc_object@@XZ"() #2 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  %call.i = call i8* @"?noexcept_func@@YAPAUobjc_object@@XZ"() #2 [ "funclet"(token %0), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  cleanupret from %0 unwind to caller
}

declare i8* @"\01?f@@YAPAUobjc_object@@XZ"()

declare i32 @__CxxFrameHandler3(...)

declare void @"?foo@@YAXXZ"()
declare i8* @"?noexcept_func@@YAPAUobjc_object@@XZ"()

declare dllimport i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)

declare dllimport void @llvm.objc.release(i8*)

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"movl\09%ebp, %ebp\09\09// marker for objc_retainAutoreleaseReturnValue"}
