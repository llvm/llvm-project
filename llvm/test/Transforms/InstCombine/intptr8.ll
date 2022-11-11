; RUN: opt < %s  -passes=instcombine -S | FileCheck %s

; PR45033: Don't try to insert a cast into a catchswich block.

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%struct.intrusive_ptr = type { ptr }
%struct.C = type { %struct.intrusive_ref_counter }
%struct.intrusive_ref_counter = type { i32 }

declare dso_local ptr @"?mk@@YAPEAUC@@XZ"() #3
declare dso_local void @"?intrusive_ptr_release@@YAXPEBUintrusive_ref_counter@@@Z"(ptr) #3
declare dso_local void @"?terminate@@YAXXZ"()
declare dso_local i32 @__CxxFrameHandler3(...)

define dso_local void @"?crash@@YAXXZ"() local_unnamed_addr #0 personality ptr @__CxxFrameHandler3 {
entry:
  %call1 = invoke ptr @"?mk@@YAPEAUC@@XZ"()
          to label %invoke.cont2 unwind label %catch.dispatch

invoke.cont2:                                     ; preds = %entry
  %0 = ptrtoint ptr %call1 to i64
  %call5 = invoke ptr @"?mk@@YAPEAUC@@XZ"()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont2, %entry
  %a.sroa.0.0 = phi i64 [ %0, %invoke.cont2 ], [ 0, %entry ]
  %1 = catchswitch within none [label %catch] unwind label %ehcleanup

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr null, i32 64, ptr null]
  catchret from %2 to label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %catch
  %a.sroa.0.1 = phi i64 [ %0, %invoke.cont2 ], [ %a.sroa.0.0, %catch ]
  %3 = inttoptr i64 %a.sroa.0.1 to ptr
  %tobool.i3 = icmp eq ptr %3, null
  br i1 %tobool.i3, label %"??1?$intrusive_ptr@UC@@@@QEAA@XZ.exit6", label %if.then.i4

if.then.i4:                                       ; preds = %try.cont
  invoke void @"?intrusive_ptr_release@@YAXPEBUintrusive_ref_counter@@@Z"(ptr %3)
          to label %"??1?$intrusive_ptr@UC@@@@QEAA@XZ.exit6" unwind label %terminate.i5

terminate.i5:                                     ; preds = %if.then.i4
  %4 = cleanuppad within none []
  call void @"?terminate@@YAXXZ"() #4 [ "funclet"(token %4) ]
  unreachable

"??1?$intrusive_ptr@UC@@@@QEAA@XZ.exit6":         ; preds = %try.cont, %if.then.i4
  ret void

ehcleanup:                                        ; preds = %catch.dispatch
  %5 = cleanuppad within none []
  %6 = inttoptr i64 %a.sroa.0.0 to ptr
  %tobool.i = icmp eq ptr %6, null
  br i1 %tobool.i, label %"??1?$intrusive_ptr@UC@@@@QEAA@XZ.exit", label %if.then.i

if.then.i:                                        ; preds = %ehcleanup
  invoke void @"?intrusive_ptr_release@@YAXPEBUintrusive_ref_counter@@@Z"(ptr %6) [ "funclet"(token %5) ]
          to label %"??1?$intrusive_ptr@UC@@@@QEAA@XZ.exit" unwind label %terminate.i

terminate.i:                                      ; preds = %if.then.i
  %7 = cleanuppad within %5 []
  call void @"?terminate@@YAXXZ"() #4 [ "funclet"(token %7) ]
  unreachable

"??1?$intrusive_ptr@UC@@@@QEAA@XZ.exit":          ; preds = %ehcleanup, %if.then.i
  cleanupret from %5 unwind to caller
}

; CHECK-LABEL: define dso_local void @"?crash@@YAXXZ"
; CHECK: catch.dispatch:
; CHECK-NEXT: %a.sroa.0.0 = phi i64
; CHECK-NEXT: catchswitch within none [label %catch] unwind label %ehcleanup
