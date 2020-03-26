; RUN: opt < %s -enable-coroutines -O2 -S | FileCheck %s
target datalayout = "E-p:32:32"

define i8* @f(i8* %buffer, i32 %n, i8** swifterror %errorslot) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 8, i32 4, i8* %buffer, i8* bitcast (i8* (i8*, i1, i8**)* @f_prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %resume ]
  call void @print(i32 %n.val)
  call void @maybeThrow(i8** swifterror %errorslot)
  %errorload1 = load i8*, i8** %errorslot
  call void @logError(i8* %errorload1)
  %suspend_result = call { i1, i8** } (...) @llvm.coro.suspend.retcon.i1p0p0i8()
  %unwind0 = extractvalue { i1, i8** } %suspend_result, 0
  br i1 %unwind0, label %cleanup, label %resume

resume:
  %inc = add i32 %n.val, 1
  br label %loop

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define i8* @f(i8* %buffer, i32 %n, i8** swifterror %errorslot)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T0]], align 4
; CHECK-NEXT:    call void @print(i32 %n)
;   TODO: figure out a way to eliminate this
; CHECK-NEXT:    store i8* null, i8** %errorslot
; CHECK-NEXT:    call void @maybeThrow(i8** nonnull swifterror %errorslot)
; CHECK-NEXT:    [[T1:%.*]] = load i8*, i8** %errorslot
; CHECK-NEXT:    call void @logError(i8* [[T1]])
; CHECK-NEXT:    store i8* [[T1]], i8** %errorslot
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i1, i8**)* @f.resume.0 to i8*)
; CHECK-NEXT:  }

; CHECK-LABEL: define internal i8* @f.resume.0(i8* noalias nonnull align 4 dereferenceable(8) %0, i1 zeroext %1, i8** swifterror %2)
; CHECK-NEXT:  :
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[ERROR:%.*]] = load i8*, i8** %2, align 4
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to i32*
; CHECK-NEXT:    [[T1:%.*]] = load i32, i32* [[T0]], align 4
; CHECK-NEXT:    %inc = add i32 [[T1]], 1
; CHECK-NEXT:    store i32 %inc, i32* [[T0]], align 4
; CHECK-NEXT:    call void @print(i32 %inc)
; CHECK-NEXT:    store i8* [[ERROR]], i8** %2
; CHECK-NEXT:    call void @maybeThrow(i8** nonnull swifterror %2)
; CHECK-NEXT:    [[T2:%.*]] = load i8*, i8** %2
; CHECK-NEXT:    call void @logError(i8* [[T2]])
; CHECK-NEXT:    store i8* [[T2]], i8** %2
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i1, i8**)* @f.resume.0 to i8*)
; CHECK:       :
; CHECK-NEXT:    ret i8* null
; CHECK-NEXT:  }

define i8* @g(i8* %buffer, i32 %n) {
entry:
  %errorslot = alloca swifterror i8*, align 4
  store i8* null, i8** %errorslot
  %id = call token @llvm.coro.id.retcon(i32 8, i32 4, i8* %buffer, i8* bitcast (i8* (i8*, i1)* @g_prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %resume ]
  call void @print(i32 %n.val)
  call void @maybeThrow(i8** swifterror %errorslot)
  %errorload1 = load i8*, i8** %errorslot
  call void @logError(i8* %errorload1)
  %unwind0 = call i1 (...) @llvm.coro.suspend.retcon.i1()
  br i1 %unwind0, label %cleanup, label %resume

resume:
  %inc = add i32 %n.val, 1
  br label %loop

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define i8* @g(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ERRORSLOT:%.*]] = alloca swifterror i8*, align 4
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T0]], align 4
; CHECK-NEXT:    call void @print(i32 %n)
; CHECK-NEXT:    store i8* null, i8** [[ERRORSLOT]], align 4
; CHECK-NEXT:    call void @maybeThrow(i8** nonnull swifterror [[ERRORSLOT]])
; CHECK-NEXT:    [[T1:%.*]] = load i8*, i8** [[ERRORSLOT]], align 4
; CHECK-NEXT:    [[T2:%.*]] = getelementptr inbounds i8, i8* %buffer, i32 4
; CHECK-NEXT:    [[T3:%.*]] = bitcast i8* [[T2]] to i8**
; CHECK-NEXT:    store i8* [[T1]], i8** [[T3]], align 4
; CHECK-NEXT:    call void @logError(i8* [[T1]])
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i1)* @g.resume.0 to i8*)
; CHECK-NEXT:  }

; CHECK-LABEL: define internal i8* @g.resume.0(i8* noalias nonnull align 4 dereferenceable(8) %0, i1 zeroext %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[ERRORSLOT:%.*]] = alloca swifterror i8*, align 4
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to i32*
; CHECK-NEXT:    [[T1:%.*]] = load i32, i32* [[T0]], align 4
; CHECK-NEXT:    %inc = add i32 [[T1]], 1
; CHECK-NEXT:    [[T2:%.*]] = getelementptr inbounds i8, i8* %0, i32 4
; CHECK-NEXT:    [[T3:%.*]] = bitcast i8* [[T2]] to i8**
; CHECK-NEXT:    [[T4:%.*]] = load i8*, i8** [[T3]]
; CHECK-NEXT:    store i32 %inc, i32* [[T0]], align 4
; CHECK-NEXT:    call void @print(i32 %inc)
; CHECK-NEXT:    store i8* [[T4]], i8** [[ERRORSLOT]]
; CHECK-NEXT:    call void @maybeThrow(i8** nonnull swifterror [[ERRORSLOT]])
; CHECK-NEXT:    [[T5:%.*]] = load i8*, i8** [[ERRORSLOT]]
; CHECK-NEXT:    store i8* [[T5]], i8** [[T3]], align 4
; CHECK-NEXT:    call void @logError(i8* [[T5]])
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i1)* @g.resume.0 to i8*)
; CHECK:       :
; CHECK-NEXT:    ret i8* null
; CHECK-NEXT:  }

declare token @llvm.coro.id.retcon(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare { i1, i8** } @llvm.coro.suspend.retcon.i1p0p0i8(...)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)

declare i8* @f_prototype(i8*, i1 zeroext, i8** swifterror)
declare i8* @g_prototype(i8*, i1 zeroext)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)

declare void @print(i32)
declare void @maybeThrow(i8** swifterror)
declare void @logError(i8*)
