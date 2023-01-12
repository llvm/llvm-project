; Check that a remat that inserts rematerialized instructions in the single predecessor block works
; as expected
; RUN: opt < %s -O0 -S | FileCheck %s

; CHECK: %f.Frame = type { i32, i32 }

define { i8*, i32 } @f(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 8, i32 4, i8* %buffer, i8* bitcast ({ i8*, i32 } (i8*, i1)* @f_prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %resume1 ]
  call void @print(i32 %n.val)
  %inc1 = add i32 %n.val, 1
  %inc2 = add i32 %inc1, 2
  %inc3 = add i32 %inc2, 3
  %inc4 = add i32 %inc3, 4
  %inc5 = add i32 %inc4, 5
  %inc6 = add i32 %inc5, 6
  %unwind0 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %inc6)
  br i1 %unwind0, label %cleanup, label %resume

resume:
  %unwind1 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %inc6)
  br i1 %unwind1, label %cleanup, label %resume1

resume1:
  %inc = add i32 %n.val, 1
  br label %loop

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

declare token @llvm.coro.id.retcon(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)

declare { i8*, i32 } @f_prototype(i8*, i1 zeroext)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)

declare void @print(i32)
