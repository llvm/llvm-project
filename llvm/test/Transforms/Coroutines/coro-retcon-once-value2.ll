; RUN: opt < %s -coro-split -coro-cleanup -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define {i8*, i32*} @f(i8* %buffer, i32* %ptr) "coroutine.presplit"="1" {
entry:
  %temp = alloca i32, align 4
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, i8* %buffer, i8* bitcast (void (i8*, i1)* @prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %oldvalue = load i32, i32* %ptr
  store i32 %oldvalue, i32* %temp
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32* %temp)
  br i1 %unwind, label %cleanup, label %cont

cont:
  %newvalue = load i32, i32* %temp
  store i32 %newvalue, i32* %ptr
  br label %cleanup

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define { i8*, i32* } @f(i8* %buffer, i32* %ptr)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALLOC:%.*]] = call i8* @allocate(i32 16)
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i8**
; CHECK-NEXT:    store i8* [[ALLOC]], i8** [[T0]]
; CHECK-NEXT:    [[FRAME:%.*]] = bitcast i8* [[ALLOC]] to [[FRAME_T:%.*]]*
; CHECK-NEXT:    %temp = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 1
; CHECK-NEXT:    [[SPILL:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 0
; CHECK-NEXT:    store i32* %ptr, i32** [[SPILL]]
; CHECK-NEXT:    %oldvalue = load i32, i32* %ptr
; CHECK-NEXT:    store i32 %oldvalue, i32* %temp
; CHECK-NEXT:    [[T0:%.*]] = insertvalue { i8*, i32* } { i8* bitcast (void (i8*, i1)* @f.resume.0 to i8*), i32* undef }, i32* %temp, 1
; CHECK-NEXT:    ret { i8*, i32* } [[T0]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal void @f.resume.0(i8* noalias nonnull align 8 dereferenceable(8) %0, i1 zeroext %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to [[FRAME_T:%.*]]**
; CHECK-NEXT:    [[FRAME:%.*]] = load [[FRAME_T]]*, [[FRAME_T]]** [[T0]]
; CHECK-NEXT:    bitcast [[FRAME_T]]* [[FRAME]] to i8*
; CHECK-NEXT:    %temp = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 1
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[TEMP_SLOT:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 1
; CHECK-NEXT:    [[PTR_SLOT:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 0
; CHECK-NEXT:    [[PTR_RELOAD:%.*]] = load i32*, i32** [[PTR_SLOT]]
; CHECK-NEXT:    %newvalue = load i32, i32* [[TEMP_SLOT]]
; CHECK-NEXT:    store i32 %newvalue, i32* [[PTR_RELOAD]]
; CHECK-NEXT:    br label
; CHECK:       :
; CHECK-NEXT:    [[T0:%.*]] = bitcast [[FRAME_T]]* [[FRAME]] to i8*
; CHECK-NEXT:    call fastcc void @deallocate(i8* [[T0]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(i8*, i1)

declare void @prototype(i8*, i1 zeroext)

declare noalias i8* @allocate(i32 %size)
declare fastcc void @deallocate(i8* %ptr)

declare void @print(i32)

