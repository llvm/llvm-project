; RUN: opt -passes='module(coro-early),cgscc(coro-split)' -S < %s | FileCheck %s
; RUN: opt -passes='module(coro-early),cgscc(coro-split)' -opaque-pointers=1 -S < %s | FileCheck %s

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)

declare ptr @llvm.coro.begin(token, ptr writeonly)

declare token @llvm.coro.save(ptr)

declare i8 @llvm.coro.suspend(token, i1)

declare i1 @get.i1()

; CHECK-LABEL: define void @foo()
; CHECK-LABEL: define {{.*}}void @foo.resume(
; CHECK: call void @foo()
; CHECK-LABEL: define {{.*}}void @foo.destroy(

define void @foo() presplitcoroutine {
entry:
  %__promise = alloca i32, align 8
  %0 = call token @llvm.coro.id(i32 16, ptr %__promise, ptr null, ptr null)
  %1 = call ptr @llvm.coro.begin(token %0, ptr null)
  %c = call i1 @get.i1()
  br i1 %c, label %if.then154, label %init.suspend

init.suspend:                                     ; preds = %entry
  %save = call token @llvm.coro.save(ptr null)
  %i3 = call i8 @llvm.coro.suspend(token %save, i1 false)
  %cond = icmp eq i8 %i3, 0
  br i1 %cond, label %if.then154, label %invoke.cont163

if.then154:                                       ; preds = %init.suspend, %entry
  call void @foo()
  br label %invoke.cont163

invoke.cont163:                                   ; preds = %if.then154, %init.suspend
  ret void
}
