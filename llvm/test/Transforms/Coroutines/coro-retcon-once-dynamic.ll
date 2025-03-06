; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),module(coro-cleanup)' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm64-apple-macos99.99"

; CHECK-LABEL: %func.Frame = type { ptr }

; CHECK-LABEL: @func_cfp = constant <{ i32, i32 }> 
; CHECK-SAME:  <{ 
; CHECK-SAME:    i32 trunc
; CHECK-SAME:    i32 16
; CHECK-SAME:  }>
@func_cfp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; offset to @func from @func_cfp
       i64 sub (
         i64 ptrtoint (ptr @func to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @func_cfp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 64 ; frame size
}>


; CHECK-LABEL: @func(
; CHECK-SAME:      ptr %buffer,
; CHECK-SAME:      ptr %allocator
; CHECK-SAME:      ptr %array
; CHECK-SAME:  ) {
; CHECK:           %array.spill.addr = getelementptr inbounds %func.Frame, ptr %buffer, i32 0, i32 0
; CHECK:           store ptr %array, ptr %array.spill.addr
; CHECK:           %load = load i32, ptr %array
; CHECK:           %load.positive = icmp sgt i32 %load, 0
; CHECK:           [[CONTINUATION:%.*]] = select i1 %load.positive
; CHECK-SAME:          ptr @func.resume.0
; CHECK-SAME:          ptr @func.resume.1
; CHECK:           [[RETVAL_1:%.*]] = insertvalue { ptr, i32 } poison, ptr [[CONTINUATION:%.*]], 0
; CHECK:           [[RETVAL_2:%.*]] = insertvalue { ptr, i32 } [[RETVAL_1:%.*]], i32 %load, 1
; CHECK:           [[DONT_POP:%.*]] = icmp eq ptr %allocator, null
; CHECK:           br i1 [[DONT_POP:%[^,]+]],
; CHECK-SAME:          label %coro.return.popless
; CHECK-SAME:          label %coro.return.normal
; CHECK:         coro.return.popless:
; CHECK:           musttail call void @llvm.ret.popless()
; CHECK:           ret { ptr, i32 } [[RETVAL_2:%.*]]
; CHECK:         coro.return.normal:
; CHECK:           ret { ptr, i32 } [[RETVAL_2:%.*]]
; CHECK:       }

; CHECK-LABEL: @func.resume.0(
; CHECK-SAME:      ptr [[BUFFER:[^,]+]]
; CHECK-SAME:      ptr [[ALLOCATOR:%[^)]+]]
; CHECK-SAME:  ) {
; CHECK:           %array.reload.addr3 = getelementptr inbounds %func.Frame, ptr [[BUFFER:%.*]], i32 0, i32 0
; CHECK:           %array.reload4 = load ptr, ptr %array.reload.addr3
; CHECK:           store i32 0, ptr %array.reload4
; CHECK:           ret void
; CHECK:       }

; CHECK-LABEL: @func.resume.1(
; CHECK-SAME:      ptr [[BUFFER:[^,]+]]
; CHECK-SAME:      ptr [[ALLOCATOR:%[^)]+]]
; CHECK-SAME:  ) {
; CHECK:           %array.reload.addr = getelementptr inbounds %func.Frame, ptr [[BUFFER:%.*]], i32 0, i32 0
; CHECK:           %array.reload = load ptr, ptr %array.reload.addr
; CHECK:           store i32 10, ptr %array.reload
; CHECK:           ret void
; CHECK:       }
define swiftcorocc {ptr, i32} @func(ptr %buffer, ptr %allocator, ptr %array) {
entry:
  %id = call token @llvm.coro.id.retcon.once.dynamic(
    i32 -1, 
    i32 16, 
    ptr @func_cfp, 
    ptr %allocator, 
    ptr %buffer, 
    ptr @continuation_prototype, 
    ptr @allocate, 
    ptr @deallocate
  )
  %handle = call ptr @llvm.coro.begin(token %id, ptr null)
  %load = load i32, ptr %array
  %load.positive = icmp sgt i32 %load, 0
  br i1 %load.positive, label %positive, label %negative

positive:
  call ptr (...) @llvm.coro.suspend.retcon.p0(i32 %load)
  store i32 0, ptr %array, align 4
  br label %cleanup

negative:
  call ptr (...) @llvm.coro.suspend.retcon.p0(i32 %load)
  store i32 10, ptr %array, align 4
  br label %cleanup

cleanup:
  call i1 @llvm.coro.end(ptr %handle, i1 0, token none)
  unreachable
}

declare void @continuation_prototype(ptr, ptr)

declare swiftcorocc noalias ptr @allocate(i32 %size)
declare void @deallocate(ptr %ptr)
