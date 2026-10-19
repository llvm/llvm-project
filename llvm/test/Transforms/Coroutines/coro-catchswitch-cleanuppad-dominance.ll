; Tests that values flowing into the PHI nodes of a cleanuppad with a
; catchswitch predecessor stay valid when the PHI nodes are split up through a
; dispatcher block, also if they do not dominate the cleanuppad
; (https://github.com/llvm/llvm-project/issues/226735).
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s --check-prefix=RESUME

declare i32 @__CxxFrameHandler3(...)

; %val is a PHI in the catchswitch block, so it does not dominate the
; dispatcher, which is also reached from %entry. It is passed through a PHI in
; the dispatcher.

; CHECK-LABEL: define ptr @catchswitch_phi(
; CHECK:       dispatch:
; CHECK-NEXT:    %val = phi i32
; CHECK-NEXT:    %cs = catchswitch within none [label %catch] unwind label %outer.cleanup.corodispatch
; CHECK:       outer.cleanup.corodispatch:
; CHECK-NEXT:    [[SWITCHVAL:%.*]] = phi i8 [ 0, %dispatch ], [ 1, %PostSpill ]
; CHECK-NEXT:    %val.corodispatch = phi i32 [ %val, %dispatch ], [ poison, %PostSpill ]
; CHECK-NEXT:    %cp3 = cleanuppad within none []
; CHECK-NEXT:    switch i8 [[SWITCHVAL]], label %unreachable [
; CHECK-NEXT:      i8 0, label %outer.cleanup.from.dispatch
; CHECK-NEXT:      i8 1, label %outer.cleanup.from.entry
; CHECK:       outer.cleanup.from.dispatch:
; CHECK-NEXT:    %val.outer.cleanup = phi i32 [ %val.corodispatch, %outer.cleanup.corodispatch ]
; CHECK:       outer.cleanup.from.entry:
; CHECK-NEXT:    %.outer.cleanup = phi i32 [ 0, %outer.cleanup.corodispatch ]
; CHECK:       outer.cleanup:
; CHECK-NEXT:    %val2 = phi i32 [ %.outer.cleanup, %outer.cleanup.from.entry ], [ %val.outer.cleanup, %outer.cleanup.from.dispatch ]

define ptr @catchswitch_phi() presplitcoroutine personality ptr @__CxxFrameHandler3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @catchswitch_phi, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %a = invoke i32 @f()
          to label %cont unwind label %outer.cleanup

cont:
  %b = invoke i32 @f()
          to label %cont2 unwind label %cleanup1

cont2:
  %c = invoke i32 @f()
          to label %suspend.point unwind label %cleanup2

suspend.point:
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s, label %suspend [i8 0, label %cleanup
                                i8 1, label %cleanup]

cleanup1:
  %cp1 = cleanuppad within none []
  cleanupret from %cp1 unwind label %dispatch

cleanup2:
  %cp2 = cleanuppad within none []
  cleanupret from %cp2 unwind label %dispatch

dispatch:
  %val = phi i32 [ %a, %cleanup1 ], [ %b, %cleanup2 ]
  %cs = catchswitch within none [label %catch] unwind label %outer.cleanup

catch:
  %catchpad = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %catchpad to label %cleanup

outer.cleanup:
  %val2 = phi i32 [ 0, %entry ], [ %val, %dispatch ]
  %cp3 = cleanuppad within none []
  call void @print(i32 %val2) [ "funclet"(token %cp3) ]
  cleanupret from %cp3 unwind to caller

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

; %val is the result of an invoke that unwinds to the cleanuppad, so it does
; not dominate the cleanuppad either.

; CHECK-LABEL: define ptr @invoke_result(
; CHECK:       outer.cleanup.corodispatch:
; CHECK-NEXT:    [[SWITCHVAL:%.*]] = phi i8 [ 0, %dispatch ], [ 1, %PostSpill ]
; CHECK-NEXT:    %val.corodispatch = phi i32 [ %val, %dispatch ], [ poison, %PostSpill ]
; CHECK-NEXT:    %cp = cleanuppad within none []
; CHECK:       outer.cleanup.from.dispatch:
; CHECK-NEXT:    %val.outer.cleanup = phi i32 [ %val.corodispatch, %outer.cleanup.corodispatch ]

define ptr @invoke_result() presplitcoroutine personality ptr @__CxxFrameHandler3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @invoke_result, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %val = invoke i32 @f()
          to label %try unwind label %outer.cleanup

try:
  invoke void @print(i32 1)
          to label %suspend.point unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind label %outer.cleanup

catch:
  %catchpad = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %catchpad to label %suspend.point

suspend.point:
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s, label %suspend [i8 0, label %cleanup
                                i8 1, label %cleanup]

outer.cleanup:
  %val2 = phi i32 [ 0, %entry ], [ %val, %dispatch ]
  %cp = cleanuppad within none []
  call void @print(i32 %val2) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

; %val does not dominate the cleanuppad, but it lives across the suspend point,
; so it must be spilled and reloaded rather than passed through the
; dispatcher, which is not analyzed for suspend crossing.

; CHECK-LABEL: define ptr @non_dominating_crossing(
; CHECK:         %val = call i32 @f()
; CHECK-NEXT:    %val.spill.addr = getelementptr inbounds i8, ptr %hdl, i64 16
; CHECK-NEXT:    store i32 %val, ptr %val.spill.addr, align 4
; CHECK:       outer.cleanup.corodispatch:
; CHECK-NOT:     corodispatch = phi
; CHECK:         cleanuppad within none []
; CHECK:       outer.cleanup.from.dispatch:
; CHECK-NEXT:    %val.reload.addr = getelementptr inbounds i8, ptr %hdl, i64 16
; CHECK-NEXT:    %val.reload = load i32, ptr %val.reload.addr, align 4
; CHECK:       outer.cleanup:
; CHECK-NEXT:    %val2 = phi i32 [ %.outer.cleanup, %outer.cleanup.from.entry ], [ %val.reload, %outer.cleanup.from.dispatch ]

; RESUME-LABEL: define internal void @non_dominating_crossing.resume(
; RESUME:       outer.cleanup.from.dispatch:
; RESUME-NEXT:    %val.reload.addr = getelementptr inbounds i8, ptr %hdl, i64 16
; RESUME-NEXT:    %val.reload = load i32, ptr %val.reload.addr, align 4
; RESUME:       outer.cleanup:
; RESUME-NEXT:    %val2 = phi i32 [ %.outer.cleanup, %outer.cleanup.from.entry ], [ %val.reload, %outer.cleanup.from.dispatch ]

define ptr @non_dominating_crossing() presplitcoroutine personality ptr @__CxxFrameHandler3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @non_dominating_crossing, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  invoke void @print(i32 0)
          to label %cont unwind label %outer.cleanup

cont:
  %val = call i32 @f()
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s, label %suspend [i8 0, label %try
                                i8 1, label %cleanup]

try:
  invoke void @print(i32 1)
          to label %cleanup unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind label %outer.cleanup

catch:
  %catchpad = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %catchpad to label %cleanup

outer.cleanup:
  %val2 = phi i32 [ 0, %entry ], [ %val, %dispatch ]
  %cp = cleanuppad within none []
  call void @print(i32 %val2) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

; As above, but %val only lives across the suspend point if %b. It is still
; reloaded from the frame, so it has to be stored there before the branch, on
; every path: the ramp reaches the cleanup without suspending if !%b.

; CHECK-LABEL: define ptr @conditional_crossing(
; CHECK:       cont:
; CHECK-NEXT:    %val = call i32 @f()
; CHECK-NEXT:    %val.spill.addr = getelementptr inbounds i8, ptr %hdl, i64 16
; CHECK-NEXT:    store i32 %val, ptr %val.spill.addr, align 4
; CHECK-NEXT:    br i1 %b,
; CHECK:       outer.cleanup.corodispatch:
; CHECK-NOT:     corodispatch = phi
; CHECK:         cleanuppad within none []
; CHECK:       outer.cleanup.from.dispatch:
; CHECK-NEXT:    %val.reload.addr = getelementptr inbounds i8, ptr %hdl, i64 16
; CHECK-NEXT:    %val.reload = load i32, ptr %val.reload.addr, align 4
; CHECK:       outer.cleanup:
; CHECK-NEXT:    %val2 = phi i32 [ %.outer.cleanup, %outer.cleanup.from.entry ], [ %val.reload, %outer.cleanup.from.dispatch ]

; RESUME-LABEL: define internal void @conditional_crossing.resume(
; RESUME:       outer.cleanup.from.dispatch:
; RESUME-NEXT:    %val.reload.addr = getelementptr inbounds i8, ptr %hdl, i64 16
; RESUME-NEXT:    %val.reload = load i32, ptr %val.reload.addr, align 4

define ptr @conditional_crossing(i1 %b) presplitcoroutine personality ptr @__CxxFrameHandler3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @conditional_crossing, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  invoke void @print(i32 0)
          to label %cont unwind label %outer.cleanup

cont:
  %val = call i32 @f()
  br i1 %b, label %suspend.point, label %try

suspend.point:
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s, label %suspend [i8 0, label %try
                                i8 1, label %cleanup]

try:
  invoke void @print(i32 1)
          to label %cleanup unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind label %outer.cleanup

catch:
  %catchpad = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %catchpad to label %cleanup

outer.cleanup:
  %val2 = phi i32 [ 0, %entry ], [ %val, %dispatch ]
  %cp = cleanuppad within none []
  call void @print(i32 %val2) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8 @llvm.coro.suspend(token, i1)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @free(ptr)
declare void @print(i32)
declare i32 @f()
