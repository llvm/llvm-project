; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),module(coro-cleanup)' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm64-apple-macos99.99"

%coro_func_pointer = type <{ i32, i32 }>

; CHECK-LABEL: %func.Frame = type { ptr }
; CHECK-LABEL: %big_types.Frame = type { <32 x i8>, [16 x i8], i64, ptr, %Integer8 }

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

@big_types_cfp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; offset to @func from @big_types_cfp
       i64 sub (
         i64 ptrtoint (ptr @big_types to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @big_types_cfp, i32 0, i32 1) to i64)
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
    ptr nonnull @allocate,
    ptr nonnull @deallocate,
    ptr nonnull @allocate_frame,
    ptr nonnull @deallocate_frame
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

declare swiftcorocc noalias ptr @allocate(ptr swiftcoro %cator, i32 %size)
declare void @deallocate(ptr swiftcoro %cator, ptr %ptr)
declare swiftcorocc noalias ptr @allocate_frame(ptr swiftcoro %cator, i32 %size)
declare void @deallocate_frame(ptr swiftcoro %cator, ptr %ptr)

declare void @use(ptr %ptr)

%Integer8 = type { i8 }

; CHECK-LABEL: @big_types(
; CHECK-SAME:      ptr noalias %frame,
; CHECK-SAME:      ptr swiftcoro %allocator,
; CHECK-SAME:      i64 %index,
; CHECK-SAME:      ptr swiftself dereferenceable(32) %vec_addr
; CHECK-SAME:  ) {
; CHECK:         [[VEC_STK_BASE_PTR:%.*]] = getelementptr inbounds %big_types.Frame, ptr %frame, i32 0, i32 0
; CHECK:         [[VEC_STK_BASE_INT:%.*]] = ptrtoint ptr [[VEC_STK_BASE_PTR]] to i64
; CHECK:         [[VEC_STK_BIASED_INT:%.*]] = add i64 [[VEC_STK_BASE_INT]], 31
; CHECK:         [[VEC_STK_ALIGNED_INT:%.*]] = and i64 [[VEC_STK_BIASED_INT]], -32
; CHECK:         %vec_stk = inttoptr i64 [[VEC_STK_ALIGNED_INT]] to ptr
define swiftcorocc { ptr, ptr } @big_types(ptr noalias %frame, ptr swiftcoro %allocator, i64 %index, ptr nocapture swiftself dereferenceable(32) %vec_addr) {
  %element_addr = alloca %Integer8, align 1
  %id = tail call token @llvm.coro.id.retcon.once.dynamic(
    i32 -1,
    i32 16,
    ptr nonnull @big_types_cfp,
    ptr %allocator,
    ptr %frame,
    ptr @continuation_prototype,
    ptr nonnull @allocate,
    ptr nonnull @deallocate,
    ptr nonnull @allocate_frame,
    ptr nonnull @deallocate_frame
  )
  %handle = tail call ptr @llvm.coro.begin(token %id, ptr null)
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %element_addr)
  %vec_original = load <32 x i8>, ptr %vec_addr, align 32
  %vec_stk = alloca <32 x i8>, align 32
  store <32 x i8> %vec_original, ptr %vec_stk, align 32
  %vec_original_2 = load <32 x i8>, ptr %vec_stk, align 32
  %index32 = trunc i64 %index to i32
  %element_original = extractelement <32 x i8> %vec_original_2, i32 %index32
  store i8 %element_original, ptr %element_addr, align 1
  call ptr (...) @llvm.coro.suspend.retcon.p0(ptr nonnull %element_addr)
  %element_modified = load i8, ptr %element_addr, align 1
  %vec_original_3 = load <32 x i8>, ptr %vec_stk, align 32
  %vec_modified = insertelement <32 x i8> %vec_original_3, i8 %element_modified, i32 %index32
  store <32 x i8> %vec_modified, ptr %vec_addr, align 32
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %element_addr)
  call i1 @llvm.coro.end(ptr %handle, i1 false, token none)
  unreachable
}

declare i32 @getSize()
@allocating_something_else_cfp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; offset to @func from @allocating_something_else_cfp
       i64 sub (
         i64 ptrtoint (ptr @allocating_something_else to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @allocating_something_else_cfp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 64 ; frame size
}>
declare { ptr, ptr } @allocating_something_else(ptr noalias %frame, ptr swiftcoro %allocator)

; CHECK-LABEL: @allocating_something(
; CHECK-SAME:    ptr noalias %frame
; CHECK-SAME:    ptr swiftcoro %allocator
; CHECK-SAME:  )
; CHECK-SAME:  {
; CHECK:         %size = call i32 @getSize()
; CHECK:         call swiftcorocc ptr @allocate(
; CHECK-SAME:      ptr %allocator,
; CHECK-SAME:      i32 %size
; CHECK-SAME:    )
; CHECK:         call swiftcorocc ptr @allocate_frame(
; CHECK-SAME:      ptr %allocator,
; CHECK-SAME:      i32
; CHECK-SAME:    )
; CHECK-LABEL: @allocating_something.resume.0(
; CHECK-SAME:    ptr %0,
; CHECK-SAME:    ptr %1
; CHECK-SAME:  )
; CHECK-SAME:  {
; CHECK:         call void @deallocate_frame(
; CHECK-SAME:      ptr %1,
; CHECK-SAME:      ptr
; CHECK-SAME:    )
; CHECK:         call void @deallocate(
; CHECK-SAME:      ptr %1,
; CHECK-SAME:      ptr
; CHECK-SAME:    )
@allocating_something_cfp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; offset to @func from @allocating_something_cfp
       i64 sub (
         i64 ptrtoint (ptr @allocating_something to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @allocating_something_cfp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 64 ; frame size
}>
define { ptr, ptr } @allocating_something(ptr noalias %frame, ptr swiftcoro %allocator) {
entry:
  %id = tail call token @llvm.coro.id.retcon.once.dynamic(
    i32 -1,
    i32 16,
    ptr nonnull @allocating_something_cfp,
    ptr %allocator,
    ptr %frame,
    ptr @continuation_prototype,
    ptr nonnull @allocate,
    ptr nonnull @deallocate,
    ptr nonnull @allocate_frame,
    ptr nonnull @deallocate_frame
  )
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %size = call i32 @getSize()
  %alloca = call token @llvm.coro.alloca.alloc.i64(i32 %size, i32 8)
  %ptr = call ptr @llvm.coro.alloca.get(token %alloca)

  %callee_frame_size = load i32, ptr getelementptr inbounds nuw (%coro_func_pointer, ptr @allocating_something_else_cfp, i32 0, i32 1), align 8
  %callee_frame_size_64 = zext i32 %callee_frame_size to i64
  %callee_frame_token = call token @llvm.coro.alloca.alloc.frame.i64(i64 %callee_frame_size_64, i32 16)
  %callee_frame = call ptr @llvm.coro.alloca.get(token %callee_frame_token)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %callee_frame)
  %allocating_something_else = call ptr @llvm.coro.prepare.retcon(ptr @allocating_something_else)
  %10 = call swiftcc { ptr, ptr } %allocating_something_else(ptr noalias %callee_frame, ptr swiftcoro %allocator)

  call void @use(ptr %ptr)
  call ptr (...) @llvm.coro.suspend.retcon.p0(ptr null)

  call void @llvm.lifetime.end.p0(i64 -1, ptr %callee_frame)
  call void @llvm.coro.alloca.free.frame(token %callee_frame_token)


  call void @use(ptr %ptr)
  call void @llvm.coro.alloca.free(token %alloca)
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  unreachable
}
