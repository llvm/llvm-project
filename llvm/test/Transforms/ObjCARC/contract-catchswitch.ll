; RUN: opt -S -objc-arc-contract < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686--windows-msvc19.11.0"

%0 = type opaque

declare i32 @__CxxFrameHandler3(...)
declare dllimport void @objc_release(i8*) local_unnamed_addr
declare dllimport i8* @objc_retain(i8* returned) local_unnamed_addr

@p = global i8* null, align 4

declare void @f() local_unnamed_addr

define void @g() local_unnamed_addr personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %tmp = load i8*, i8** @p, align 4
  %cast = bitcast i8* %tmp to %0*
  %tmp1 = tail call i8* @objc_retain(i8* %tmp) #0
  ; Split the basic block to ensure bitcast ends up in entry.split.
  br label %entry.split

entry.split:
  invoke void @f()
          to label %invoke.cont unwind label %catch.dispatch

; Dummy nested catchswitch to test looping through the dominator tree.
catch.dispatch:
  %tmp2 = catchswitch within none [label %catch] unwind label %catch.dispatch1

catch:
  %tmp3 = catchpad within %tmp2 [i8* null, i32 64, i8* null]
  catchret from %tmp3 to label %invoke.cont

catch.dispatch1:
  %tmp4 = catchswitch within none [label %catch1] unwind label %ehcleanup

catch1:
  %tmp5 = catchpad within %tmp4 [i8 *null, i32 64, i8* null]
  catchret from %tmp5 to label %invoke.cont

invoke.cont:
  %tmp6 = load i8*, i8** @p, align 4
  %cast1 = bitcast i8* %tmp6 to %0*
  %tmp7 = tail call i8* @objc_retain(i8* %tmp6) #0
  call void @objc_release(i8* %tmp) #0, !clang.imprecise_release !0
  ; Split the basic block to ensure bitcast ends up in invoke.cont.split.
  br label %invoke.cont.split

invoke.cont.split:
  invoke void @f()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:
  ret void

ehcleanup:
  %tmp8 = phi %0* [ %cast, %catch.dispatch1 ], [ %cast1, %invoke.cont.split ]
  %tmp9 = cleanuppad within none []
  %tmp10 = bitcast %0* %tmp8 to i8*
  call void @objc_release(i8* %tmp10) #0 [ "funclet"(token %tmp9) ]
  cleanupret from %tmp9 unwind to caller
}

; CHECK-LABEL: entry.split:
; CHECK-NEXT:    %0 = bitcast i8* %tmp1 to %0*
; CHECK-NEXT:    invoke void @f()
; CHECK-NEXT:            to label %invoke.cont unwind label %catch.dispatch

; CHECK-LABEL: invoke.cont.split:
; CHECK-NEXT:    %1 = bitcast i8* %tmp7 to %0*
; CHECK-NEXT:    invoke void @f()
; CHECK-NEXT:            to label %invoke.cont1 unwind label %ehcleanup

; CHECK-LABEL: ehcleanup:
; CHECK-NEXT:    %tmp8 = phi %0* [ %0, %catch.dispatch1 ], [ %1, %invoke.cont.split ]

attributes #0 = { nounwind }

!0 = !{}
