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

declare i8* @"\01?f@@YAPAUobjc_object@@XZ"()

declare i32 @__CxxFrameHandler3(...)

declare dllimport i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)

declare dllimport void @llvm.objc.release(i8*)

!clang.arc.retainAutoreleasedReturnValueMarker = !{!0}
!0 = !{!"movl\09%ebp, %ebp\09\09// marker for objc_retainAutoreleaseReturnValue"}

; CHECK-LABEL: catch
; CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""() [ "funclet"(token %1) ]

; CHECK-LABEL: catch.1
; CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""() [ "funclet"(token %1) ]

; CHECK-LABEL: invoke.cont
; CHECK: call void asm sideeffect "movl{{.*}}%ebp, %ebp{{.*}}", ""(){{$}}
