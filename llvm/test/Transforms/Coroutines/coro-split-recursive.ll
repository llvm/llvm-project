; RUN: opt -passes='default<O2>' -enable-coroutines -S < %s | FileCheck %s

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)

declare i8* @llvm.coro.begin(token, i8* writeonly)

declare token @llvm.coro.save(i8*)

declare i8 @llvm.coro.suspend(token, i1)

; CHECK-LABEL: define void @foo()
; CHECK-LABEL: define {{.*}}void @foo.resume(
; CHECK: call void @foo()
; CHECK-LABEL: define {{.*}}void @foo.destroy(

define void @foo() {
entry:
  %__promise = alloca i32, align 8
  %0 = bitcast i32* %__promise to i8*
  %1 = call token @llvm.coro.id(i32 16, i8* %0, i8* null, i8* null)
  %2 = call i8* @llvm.coro.begin(token %1, i8* null)
  br i1 undef, label %if.then154, label %init.suspend

init.suspend:                                     ; preds = %entry
  %save = call token @llvm.coro.save(i8* null)
  %3 = call i8 @llvm.coro.suspend(token %save, i1 false)
  %cond = icmp eq i8 %3, 0
  br i1 %cond, label %if.then154, label %invoke.cont163

if.then154:                                       ; preds = %init.suspend, %entry
  call void @foo()
  br label %invoke.cont163

invoke.cont163:                                   ; preds = %if.then154, %init.suspend
  ret void
}
