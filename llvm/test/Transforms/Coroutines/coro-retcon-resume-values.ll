; RUN: opt < %s -enable-coroutines -O2 -S | FileCheck %s

define i8* @f(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 8, i32 4, i8* %buffer, i8* bitcast (i8* (i8*, i32, i1)* @prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %sum, %resume ]
  %values = call { i32, i1 } (...) @llvm.coro.suspend.retcon.sl_i32i1s()
  %finished = extractvalue { i32, i1 } %values, 1
  br i1 %finished, label %cleanup, label %resume

resume:
  %input = extractvalue { i32, i1 } %values, 0
  %sum = add i32 %n.val, %input
  br label %loop

cleanup:
  call void @print(i32 %n.val)
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define i8* @f(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T0]], align 4
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i32, i1)* @f.resume.0 to i8*)
; CHECK-NEXT:  }

; CHECK-LABEL: define internal i8* @f.resume.0(i8* noalias nonnull align 4 dereferenceable(8) %0, i32 %1, i1 zeroext %2)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to i32*
; CHECK-NEXT:    [[T1:%.*]] = load i32, i32* [[T0]], align 4
; CHECK-NEXT:    br i1 %2,
; CHECK:       :
; CHECK-NEXT:    %sum = add i32 [[T1]], %1
; CHECK-NEXT:    store i32 %sum, i32* [[T0]], align 4
; CHECK-NEXT:    ret i8* bitcast (i8* (i8*, i32, i1)* @f.resume.0 to i8*)
; CHECK:       :
; CHECK-NEXT:    call void @print(i32 [[T1]])
; CHECK-NEXT:    ret i8* null
; CHECK-NEXT:  }

define i32 @main() {
entry:
  %0 = alloca [8 x i8], align 4
  %buffer = bitcast [8 x i8]* %0 to i8*
  %prepare = call i8* @llvm.coro.prepare.retcon(i8* bitcast (i8* (i8*, i32)* @f to i8*))
  %f = bitcast i8* %prepare to i8* (i8*, i32)*
  %cont0 = call i8* %f(i8* %buffer, i32 1)
  %cont0.cast = bitcast i8* %cont0 to i8* (i8*, i32, i1)*
  %cont1 = call i8* %cont0.cast(i8* %buffer, i32 2, i1 zeroext false)
  %cont1.cast = bitcast i8* %cont1 to i8* (i8*, i32, i1)*
  %cont2 = call i8* %cont1.cast(i8* %buffer, i32 4, i1 zeroext false)
  %cont2.cast = bitcast i8* %cont2 to i8* (i8*, i32, i1)*
  call i8* %cont2.cast(i8* %buffer, i32 100, i1 zeroext true)
  ret i32 0
}

;   Unfortunately, we don't seem to fully optimize this right now due
;   to some sort of phase-ordering thing.
; CHECK-LABEL: define i32 @main
; CHECK-NEXT:  entry:
; CHECK:         [[BUFFER:%.*]] = alloca [8 x i8], align 4
; CHECK:         [[SLOT:%.*]] = bitcast [8 x i8]* [[BUFFER]] to i32*
; CHECK-NEXT:    store i32 7, i32* [[SLOT]], align 4
; CHECK-NEXT:    call void @print(i32 7)
; CHECK-NEXT:    ret i32 0

declare token @llvm.coro.id.retcon(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare { i32, i1 } @llvm.coro.suspend.retcon.sl_i32i1s(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)

declare i8* @prototype(i8*, i32, i1 zeroext)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)

declare void @print(i32)

