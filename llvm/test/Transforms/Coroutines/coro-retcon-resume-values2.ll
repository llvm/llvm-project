; RUN: opt < %s -coro-split -coro-cleanup -S | FileCheck %s

define i8* @f(i8* %buffer, i32 %n) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id.retcon(i32 8, i32 4, i8* %buffer, i8* bitcast (i8* (i8*, i32)* @prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %value0 = call i32 (...) @llvm.coro.suspend.retcon.i32()
  %sum0 = call i32 @add(i32 %n, i32 %value0)
  %value1 = call i32 (...) @llvm.coro.suspend.retcon.i32()
  %sum1 = call i32 @add(i32 %sum0, i32 %value0)
  %sum2 = call i32 @add(i32 %sum1, i32 %value1)
  %value2 = call i32 (...) @llvm.coro.suspend.retcon.i32()
  %sum3 = call i32 @add(i32 %sum2, i32 %value0)
  %sum4 = call i32 @add(i32 %sum3, i32 %value1)
  %sum5 = call i32 @add(i32 %sum4, i32 %value2)
  call void @print(i32 %sum5)
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define i8* @f(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALLOC:%.*]] = call i8* @allocate(i32 20)
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i8**
; CHECK-NEXT:    store i8* [[ALLOC]], i8** [[T0]]
; CHECK-NEXT:    [[FRAME:%.*]] = bitcast i8* [[ALLOC]] to [[FRAME_T:%.*]]*
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 0
; CHECK-NEXT:    store i32 %n, i32* [[T0]]
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i32)* @f.resume.0 to i8*)
; CHECK-NEXT:  }

; CHECK-LABEL: define internal i8* @f.resume.0(i8* noalias nonnull align 4 dereferenceable(8) %0, i32 %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to [[FRAME_T:%.*]]**
; CHECK-NEXT:    [[FRAME:%.*]] = load [[FRAME_T]]*, [[FRAME_T]]** [[T0]]
; CHECK-NEXT:    [[VFRAME:%.*]] = bitcast [[FRAME_T]]* [[FRAME]] to i8*
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 1
; CHECK-NEXT:    store i32 %1, i32* [[T0]]
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 0
; CHECK-NEXT:    [[N:%.*]] = load i32, i32* [[T0]]
; CHECK-NEXT:    %sum0 = call i32 @add(i32 [[N]], i32 %1)
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 2
; CHECK-NEXT:    store i32 %sum0, i32* [[T0]]
; CHECK-NEXT:    [[CONT:%.*]] = bitcast i8* (i8*, i32)* @f.resume.1 to i8*
; CHECK-NEXT:    ret i8* [[CONT]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal i8* @f.resume.1(i8* noalias nonnull align 4 dereferenceable(8) %0, i32 %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to [[FRAME_T:%.*]]**
; CHECK-NEXT:    [[FRAME:%.*]] = load [[FRAME_T]]*, [[FRAME_T]]** [[T0]]
; CHECK-NEXT:    [[VFRAME:%.*]] = bitcast [[FRAME_T]]* [[FRAME]] to i8*
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 3
; CHECK-NEXT:    store i32 %1, i32* [[T0]]
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 2
; CHECK-NEXT:    [[SUM0:%.*]] = load i32, i32* [[T0]]
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 1
; CHECK-NEXT:    [[VALUE0:%.*]] = load i32, i32* [[T0]]
; CHECK-NEXT:    %sum1 = call i32 @add(i32 [[SUM0]], i32 [[VALUE0]])
; CHECK-NEXT:    %sum2 = call i32 @add(i32 %sum1, i32 %1)
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 4
; CHECK-NEXT:    store i32 %sum2, i32* [[T0]]
; CHECK-NEXT:    [[CONT:%.*]] = bitcast i8* (i8*, i32)* @f.resume.2 to i8*
; CHECK-NEXT:    ret i8* [[CONT]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal i8* @f.resume.2(i8* noalias nonnull align 4 dereferenceable(8) %0, i32 %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to [[FRAME_T:%.*]]**
; CHECK-NEXT:    [[FRAME:%.*]] = load [[FRAME_T]]*, [[FRAME_T]]** [[T0]]
; CHECK-NEXT:    [[VFRAME:%.*]] = bitcast [[FRAME_T]]* [[FRAME]] to i8*
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 4
; CHECK-NEXT:    [[SUM2:%.*]] = load i32, i32* [[T0]]
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 3
; CHECK-NEXT:    [[VALUE1:%.*]] = load i32, i32* [[T0]]
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds [[FRAME_T]], [[FRAME_T]]* [[FRAME]], i32 0, i32 1
; CHECK-NEXT:    [[VALUE0:%.*]] = load i32, i32* [[T0]]
; CHECK-NEXT:    %sum3 = call i32 @add(i32 [[SUM2]], i32 [[VALUE0]])
; CHECK-NEXT:    %sum4 = call i32 @add(i32 %sum3, i32 [[VALUE1]])
; CHECK-NEXT:    %sum5 = call i32 @add(i32 %sum4, i32 %1)
; CHECK-NEXT:    call void @print(i32 %sum5)
; CHECK-NEXT:    [[CONT:%.*]] = bitcast [[FRAME_T]]* [[FRAME]] to i8*
; CHECK-NEXT:    call void @deallocate(i8* [[CONT]])
; CHECK-NEXT:    ret i8* null
; CHECK-NEXT:  }

declare token @llvm.coro.id.retcon(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i32 @llvm.coro.suspend.retcon.i32(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)

declare i8* @prototype(i8*, i32)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)

declare i32 @add(i32, i32)
declare void @print(i32)

