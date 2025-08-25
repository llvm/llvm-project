; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split)' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm64-apple-macos99.99"


@func_cfp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (ptr @func to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @func_cfp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 64
}>


%func_int = type <{ i64 }>
%func_obj = type <{ %func_int, ptr }>
%func_guts = type <{ %func_obj }>
%func_impl = type <{ %func_guts }>
%func_self = type <{ %func_impl }>

declare swiftcorocc void @func_continuation_prototype(ptr noalias, ptr)

; CHECK-LABEL: @func.resume.0(
; CHECK-SAME:      ptr noalias %0, 
; CHECK-SAME:      ptr %1
; CHECK-SAME:  ) {
; CHECK:       coro.return.popless:
; CHECK-NEXT:    unreachable
; CHECK:       coro.return.normal:
; CHECK-NEXT:    unreachable
; CHECK:       }

define swiftcorocc { ptr, ptr } @func(ptr noalias %buffer, ptr %allocator, ptr nocapture swiftself dereferenceable(16) %2) {
entry:
  %3 = call token @llvm.coro.id.retcon.once.dynamic(
    i32 -1, 
    i32 16,
    ptr @func_cfp,
    ptr %allocator,
    ptr %buffer,
    ptr @func_continuation_prototype,
    ptr @allocate, 
    ptr @deallocate
  )
  %handle = call ptr @llvm.coro.begin(token %3, ptr null)
  %yielded = getelementptr inbounds %func_self, ptr %2, i32 0, i32 0
  call ptr (...) @llvm.coro.suspend.retcon.p0(ptr %yielded)
  br i1 false, label %unwind, label %normal

normal:
  br label %coro.end

unwind:
  br label %coro.end

coro.end:
  %8 = call i1 @llvm.coro.end(ptr %handle, i1 false, token none)
  unreachable
}

declare swiftcorocc noalias ptr @allocate(i32 %size)
declare void @deallocate(ptr %ptr)
