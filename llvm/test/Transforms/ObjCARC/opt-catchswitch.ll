; RUN: opt -S -objc-arc < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686--windows-msvc"

declare i8* @f(i8*, i8*)

declare i32 @__CxxFrameHandler3(...)

declare dllimport i8* @llvm.objc.autoreleaseReturnValue(i8* returned)
declare dllimport i8* @llvm.objc.retain(i8* returned)
declare dllimport i8* @llvm.objc.retainAutoreleasedReturnValue(i8* returned)
declare dllimport void @llvm.objc.release(i8*)

define i8* @g(i8* %p, i8* %q) local_unnamed_addr personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %0 = tail call i8* @llvm.objc.retain(i8* %p) #0
  ; the following call prevents ARC optimizer from removing the retain/release
  ; pair on %p
  %v1 = call i8* @f(i8* null, i8* null)
  %1 = tail call i8* @llvm.objc.retain(i8* %q) #0
  %call = invoke i8* @f(i8* %p, i8* %q)
          to label %invoke.cont unwind label %catch.dispatch, !clang.arc.no_objc_arc_exceptions !0

catch.dispatch:
  %2 = catchswitch within none [label %catch] unwind to caller

catch:
  %3 = catchpad within %2 [i8* null, i32 64, i8* null]
  catchret from %3 to label %cleanup

invoke.cont:
  %4 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call) #0
  br label %cleanup

cleanup:
  %retval.0 = phi i8* [ %call, %invoke.cont ], [ null, %catch ]
  tail call void @llvm.objc.release(i8* %q) #0, !clang.imprecise_release !0
  tail call void @llvm.objc.release(i8* %p) #0, !clang.imprecise_release !0
  %5 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %retval.0) #0
  ret i8* %retval.0
}

; CHECK-LABEL: entry:
; CHECK-NEXT:    %0 = tail call i8* @llvm.objc.retain(i8* %p) #0
; CHECK-NEXT:    call i8* @f(i8* null, i8* null)
; CHECK-NEXT:    %call = invoke i8* @f(i8* %p, i8* %q)
; CHECK-NEXT:            to label %invoke.cont unwind label %catch.dispatch

; CHECK-LABEL: catch.dispatch:
; CHECK-NEXT:    %1 = catchswitch within none [label %catch] unwind to caller

; CHECK-LABEL: cleanup:
; CHECK:         tail call void @llvm.objc.release(i8* %p) #0

attributes #0 = { nounwind }

!0 = !{}
