; RUN: opt -S -passes=objc-arc-contract < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686--windows-msvc19.11.0"

%0 = type opaque

declare i32 @__CxxFrameHandler3(...)
declare dllimport void @llvm.objc.release(ptr) local_unnamed_addr
declare dllimport ptr @llvm.objc.retain(ptr returned) local_unnamed_addr

@p = global ptr null, align 4

declare void @f() local_unnamed_addr

define void @g() local_unnamed_addr personality ptr @__CxxFrameHandler3 {
entry:
  %tmp = load ptr, ptr @p, align 4
  %tmp1 = tail call ptr @llvm.objc.retain(ptr %tmp) #0
  ; Split the basic block to ensure bitcast ends up in entry.split.
  br label %entry.split

entry.split:
  invoke void @f()
          to label %invoke.cont unwind label %catch.dispatch

; Dummy nested catchswitch to test looping through the dominator tree.
catch.dispatch:
  %tmp2 = catchswitch within none [label %catch] unwind label %catch.dispatch1

catch:
  %tmp3 = catchpad within %tmp2 [ptr null, i32 64, ptr null]
  catchret from %tmp3 to label %invoke.cont

catch.dispatch1:
  %tmp4 = catchswitch within none [label %catch1] unwind label %ehcleanup

catch1:
  %tmp5 = catchpad within %tmp4 [ptr null, i32 64, ptr null]
  catchret from %tmp5 to label %invoke.cont

invoke.cont:
  %tmp6 = load ptr, ptr @p, align 4
  %tmp7 = tail call ptr @llvm.objc.retain(ptr %tmp6) #0
  call void @llvm.objc.release(ptr %tmp) #0, !clang.imprecise_release !0
  ; Split the basic block to ensure bitcast ends up in invoke.cont.split.
  br label %invoke.cont.split

invoke.cont.split:
  invoke void @f()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:
  ret void

ehcleanup:
  %tmp8 = phi ptr [ %tmp, %catch.dispatch1 ], [ %tmp6, %invoke.cont.split ]
  %tmp9 = cleanuppad within none []
  call void @llvm.objc.release(ptr %tmp8) #0 [ "funclet"(token %tmp9) ]
  cleanupret from %tmp9 unwind to caller
}

; CHECK-LABEL: entry.split:
; CHECK-NEXT:    invoke void @f()
; CHECK-NEXT:            to label %invoke.cont unwind label %catch.dispatch

; CHECK-LABEL: invoke.cont.split:
; CHECK-NEXT:    invoke void @f()
; CHECK-NEXT:            to label %invoke.cont1 unwind label %ehcleanup

; CHECK-LABEL: ehcleanup:
; CHECK-NEXT:    %tmp8 = phi ptr [ %tmp1, %catch.dispatch1 ], [ %tmp7, %invoke.cont.split ]

attributes #0 = { nounwind }

!0 = !{}
