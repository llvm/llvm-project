; RUN: opt -S %s | FileCheck %s

declare i1 @llvm.coro.end(ptr, i1 immarg, token)

define i1 @used_result(ptr %frame) presplitcoroutine {
; CHECK-LABEL: @used_result(
; CHECK: call void @llvm.coro.end(ptr %frame, i1 false, token none)
; CHECK-NEXT: [[IN_RAMP:%.*]] = call i1 @llvm.coro.is_in_ramp()
; CHECK-NEXT: [[IN_RESUME:%.*]] = xor i1 [[IN_RAMP]], true
; CHECK-NEXT: ret i1 [[IN_RESUME]]
entry:
  %in.resume = call i1 @llvm.coro.end(ptr %frame, i1 false, token none)
  ret i1 %in.resume
}

define void @unused_result(ptr %frame) presplitcoroutine {
; CHECK-LABEL: @unused_result(
; CHECK: call void @llvm.coro.end(ptr %frame, i1 false, token none)
; CHECK-NEXT: ret void
entry:
  %unused = call i1 @llvm.coro.end(ptr %frame, i1 false, token none)
  ret void
}

declare void @may_throw()
declare i32 @__CxxFrameHandler3(...)

define void @funclet_bundle(ptr %frame) presplitcoroutine personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: @funclet_bundle(
; CHECK: [[CP:%.*]] = cleanuppad within none []
; CHECK-NEXT: call void @llvm.coro.end(ptr %frame, i1 true, token none) [ "funclet"(token [[CP]]) ]
; CHECK-NEXT: cleanupret from [[CP]] unwind to caller
entry:
  invoke void @may_throw()
          to label %cont unwind label %ehcleanup
cont:
  ret void
ehcleanup:
  %cp = cleanuppad within none []
  %unused = call i1 @llvm.coro.end(ptr %frame, i1 true, token none) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

declare i1 @llvm.coro.end.async(ptr, i1, ...)

define i1 @used_async_result(ptr %frame) presplitcoroutine {
; CHECK-LABEL: @used_async_result(
; CHECK: call void (ptr, i1, ...) @llvm.coro.end.async(ptr %frame, i1 false)
; CHECK-NEXT: [[ASYNC_IN_RAMP:%.*]] = call i1 @llvm.coro.is_in_ramp()
; CHECK-NEXT: [[ASYNC_IN_RESUME:%.*]] = xor i1 [[ASYNC_IN_RAMP]], true
; CHECK-NEXT: ret i1 [[ASYNC_IN_RESUME]]
entry:
  %in.resume = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %frame, i1 false)
  ret i1 %in.resume
}

define void @unused_async_result(ptr %frame) presplitcoroutine {
; CHECK-LABEL: @unused_async_result(
; CHECK: call void (ptr, i1, ...) @llvm.coro.end.async(ptr %frame, i1 false)
; CHECK-NEXT: ret void
entry:
  %unused = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %frame, i1 false)
  ret void
}

; CHECK-DAG: declare void @llvm.coro.end(ptr, i1 immarg, token)
; CHECK-DAG: declare void @llvm.coro.end.async(ptr, i1 immarg, ...)
; CHECK-DAG: declare i1 @llvm.coro.is_in_ramp()
