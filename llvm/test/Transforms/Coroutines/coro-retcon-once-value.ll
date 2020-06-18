; RUN: opt < %s -enable-coroutines -O2 -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define {i8*, i32} @f(i8* %buffer, i32* %array) {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, i8* %buffer, i8* bitcast (void (i8*, i1)* @prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %load = load i32, i32* %array
  %load.pos = icmp sgt i32 %load, 0
  br i1 %load.pos, label %pos, label %neg

pos:
  %unwind0 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %load)
  br i1 %unwind0, label %cleanup, label %pos.cont

pos.cont:
  store i32 0, i32* %array, align 4
  br label %cleanup

neg:
  %unwind1 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 0)
  br i1 %unwind1, label %cleanup, label %neg.cont

neg.cont:
  store i32 10, i32* %array, align 4
  br label %cleanup

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define { i8*, i32 } @f(i8* %buffer, i32* %array)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32**
; CHECK-NEXT:    store i32* %array, i32** [[T0]], align 8
; CHECK-NEXT:    %load = load i32, i32* %array, align 4
; CHECK-NEXT:    %load.pos = icmp sgt i32 %load, 0
; CHECK-NEXT:    [[CONT:%.*]] = select i1 %load.pos, void (i8*, i1)* @f.resume.0, void (i8*, i1)* @f.resume.1
; CHECK-NEXT:    [[VAL:%.*]] = select i1 %load.pos, i32 %load, i32 0
; CHECK-NEXT:    [[CONT_CAST:%.*]] = bitcast void (i8*, i1)* [[CONT]] to i8*
; CHECK-NEXT:    [[T0:%.*]] = insertvalue { i8*, i32 } undef, i8* [[CONT_CAST]], 0
; CHECK-NEXT:    [[T1:%.*]] = insertvalue { i8*, i32 } [[T0]], i32 [[VAL]], 1
; CHECK-NEXT:    ret { i8*, i32 } [[T1]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal void @f.resume.0(i8* noalias nonnull align 8 dereferenceable(8) %0, i1 zeroext %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to i32**
; CHECK-NEXT:    [[RELOAD:%.*]] = load i32*, i32** [[T0]], align 8
; CHECK-NEXT:    store i32 0, i32* [[RELOAD]], align 4
; CHECK-NEXT:    br label
; CHECK:       :
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

; CHECK-LABEL: define internal void @f.resume.1(i8* noalias nonnull align 8 dereferenceable(8) %0, i1 zeroext %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to i32**
; CHECK-NEXT:    [[RELOAD:%.*]] = load i32*, i32** [[T0]], align 8
; CHECK-NEXT:    store i32 10, i32* [[RELOAD]], align 4
; CHECK-NEXT:    br label
; CHECK:       :
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

define void @test(i32* %array) {
entry:
  %0 = alloca [8 x i8], align 8
  %buffer = bitcast [8 x i8]* %0 to i8*
  %prepare = call i8* @llvm.coro.prepare.retcon(i8* bitcast ({i8*, i32} (i8*, i32*)* @f to i8*))
  %f = bitcast i8* %prepare to {i8*, i32} (i8*, i32*)*
  %result = call {i8*, i32} %f(i8* %buffer, i32* %array)
  %value = extractvalue {i8*, i32} %result, 1
  call void @print(i32 %value)
  %cont = extractvalue {i8*, i32} %result, 0
  %cont.cast = bitcast i8* %cont to void (i8*, i1)*
  call void %cont.cast(i8* %buffer, i1 zeroext 0)
  ret void
}

;   Unfortunately, we don't seem to fully optimize this right now due
;   to some sort of phase-ordering thing.
; CHECK-LABEL: define void @test(i32* %array)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[BUFFER:%.*]] = alloca i32*, align 8
; CHECK-NEXT:    [[BUFFER_CAST:%.*]] = bitcast i32** [[BUFFER]] to i8*
; CHECK-NEXT:    store i32* %array, i32** [[BUFFER]], align 8
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* %array, align 4
; CHECK-NEXT:    [[LOAD_POS:%.*]] = icmp sgt i32 [[LOAD]], 0
; CHECK-NEXT:    [[CONT:%.*]] = select i1 [[LOAD_POS]], void (i8*, i1)* @f.resume.0, void (i8*, i1)* @f.resume.1
; CHECK-NEXT:    [[VAL:%.*]] = select i1 [[LOAD_POS]], i32 [[LOAD]], i32 0
; CHECK-NEXT:    call void @print(i32 [[VAL]])
; CHECK-NEXT:    call void [[CONT]](i8* nonnull [[BUFFER_CAST]], i1 zeroext false)
; CHECK-NEXT:    ret void

declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)

declare void @prototype(i8*, i1 zeroext)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)

declare void @print(i32)

