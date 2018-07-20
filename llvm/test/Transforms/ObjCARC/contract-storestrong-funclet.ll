; RUN: opt -mtriple=i686-unknown-windows-msvc -objc-arc-contract -S -o - %s | FileCheck %s

declare void @f()
declare i32 @__CxxFrameHandler3(...)
declare dllimport i8* @objc_retain(i8*)
declare dllimport i8* @objc_retainAutoreleasedReturnValue(i8*)
declare dllimport void @objc_release(i8*)

@x = external global i8*

define void @g(i8* %p) personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
  invoke void @f() to label %invoke.cont unwind label %ehcleanup

invoke.cont:
  %call = tail call i8* @objc_retain(i8* %p) nounwind
  %tmp = load i8*, i8** @x, align 4
  store i8* %call, i8** @x, align 4
  tail call void @objc_release(i8* %tmp) nounwind
  ret void

ehcleanup:
  %1 = cleanuppad within none []
  %call1 = tail call i8* @objc_retain(i8* %p) nounwind [ "funclet"(token %1) ]
  %tmp1 = load i8*, i8** @x, align 4
  store i8* %call1, i8** @x, align 4
  tail call void @objc_release(i8* %tmp1) nounwind [ "funclet"(token %1) ]
  cleanupret from %1 unwind to caller
}

; CHECK-LABEL: invoke.cont:
; CHECK: tail call void @objc_storeStrong(i8** @x, i8* %p) #0{{$}}
; CHECK: ret void

; CHECK-LABEL: ehcleanup:
; CHECK: %1 = cleanuppad within none []
; CHECK: tail call void @objc_storeStrong(i8** @x, i8* %p) #0 [ "funclet"(token %1) ]
; CHECK: cleanupret from %1 unwind to caller
