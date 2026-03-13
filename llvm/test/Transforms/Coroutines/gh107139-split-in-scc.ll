; Verify that we don't crash on mutually recursive coroutines
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s

target triple = "x86_64-redhat-linux-gnu"

; CHECK-LABEL: define void @foo
define void @foo() presplitcoroutine personality ptr null {
entry:

  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %1 = call ptr @llvm.coro.begin(token %0, ptr null)
  %2 = call token @llvm.coro.save(ptr null)
  %3 = call i8 @llvm.coro.suspend(token none, i1 false)
  %4 = call token @llvm.coro.save(ptr null)
  ; CHECK: call void @bar(ptr null, ptr null)
  call void @llvm.coro.await.suspend.void(ptr null, ptr null, ptr @bar)
  ret void
}

; CHECK-LABEL: define void @bar({{.*}})
define void @bar(ptr %0, ptr %1) {
entry:
  ; CHECK: call void @foo()
  call void @foo()
  ret void
}

; CHECK-LABEL: @foo.resume({{.*}})
; CHECK-LABEL: @foo.destroy({{.*}})
; CHECK-LABEL: @foo.cleanup({{.*}})

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #0
declare ptr @llvm.coro.begin(token, ptr writeonly) nounwind
declare token @llvm.coro.save(ptr) nomerge nounwind
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare i8 @llvm.coro.suspend(token, i1) nounwind

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
