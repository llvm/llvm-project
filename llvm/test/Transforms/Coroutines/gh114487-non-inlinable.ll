; Verify that we don't crash when eliding coro_elide_safe callsites.
; RUN: opt < %s -passes='cgscc(function<>(simplifycfg<>),function-attrs,coro-annotation-elide)'  -S | FileCheck %s

; CHECK-LABEL: define void @foo()
define void @foo() presplitcoroutine personality ptr null {
entry:
  %0 = call token @llvm.coro.save(ptr null)
  br label %branch

branch:
; Check that we still call bar() because we can't inline bar.noalloc()
; CHECK: call void @bar()
  call void @bar() coro_elide_safe
  ret void
}

; CHECK-LABEL: define void @bar()
define void @bar() personality ptr null {
entry:
  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %1 = call ptr @llvm.coro.begin(token %0, ptr null)
  %2 = call token @llvm.coro.save(ptr null)
  %3 = call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}

declare void @bar.noalloc(ptr)

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) nounwind
declare token @llvm.coro.save(ptr) nomerge nounwind
declare i8 @llvm.coro.suspend(token, i1) nounwind

