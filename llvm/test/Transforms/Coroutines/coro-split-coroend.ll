; Check that we do not lower coro.end to 'ret void' if its return value is used
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f1() presplitcoroutine {
  ; CHECK-LABEL: define internal fastcc void @f1.resume(
  ; CHECK-NOT: call void @a()
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  br label %done

done:
  %InResume = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  call void @a()
  ret ptr %hdl
}

define ptr @f2() presplitcoroutine {
  ; CHECK-LABEL: define internal fastcc void @f2.resume(
  ; CHECK-NEXT: entry.resume:
  ; CHECK-NEXT: call void @a()
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  br label %end

end:
  %InResume = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  br i1 %InResume, label %resume, label %start

resume:
  call void @a()
  br label %start

start:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret ptr %hdl
}

declare void @a()
