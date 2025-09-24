; Tests that sinked lifetime markers wouldn't provent optimization
; to convert a coro.await.suspend.handle call to a musttail call.
; The difference between this and coro-split-musttail5.ll and coro-split-musttail6.ll
; is that this contains dead instruction generated during the transformation,
; which makes the optimization harder.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s
; RUN: opt < %s -passes='pgo-instr-gen,cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define i64 @g() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %alloc.var = alloca i64
  call void @llvm.lifetime.start.p0(ptr %alloc.var)
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)

  switch i8 %suspend, label %exit [
    i8 0, label %await.suspend
    i8 1, label %exit
  ]
await.suspend:
  %save2 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.handle(ptr null, ptr null, ptr @await_suspend_function)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)

  ; These (non-trivially) dead instructions are in the way.
  %gep = getelementptr inbounds i64, ptr %alloc.var, i32 0
  %foo = ptrtoint ptr %gep to i64

  switch i8 %suspend2, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  call void @consume(ptr %alloc.var)
  call void @llvm.lifetime.end.p0(ptr %alloc.var)
  br label %exit
exit:
  %result = phi i64 [0, %entry], [0, %entry], [%foo, %await.suspend], [%foo, %await.suspend], [%foo, %await.ready]
  call void @llvm.coro.end(ptr null, i1 false, token none)
  ret i64 %result
}

; Verify that in the resume part resume call is marked with musttail.
; CHECK-LABEL: @g.resume(
; CHECK:         %[[FRAME:[0-9]+]] = call ptr @await_suspend_function(ptr null, ptr null)
; CHECK:         %[[RESUMEADDR:[0-9]+]] = call ptr @llvm.coro.subfn.addr(ptr %[[FRAME]], i8 0)
; CHECK:         musttail call fastcc void %[[RESUMEADDR]](ptr %[[FRAME]])
; CHECK-NEXT:    ret void

; It has a cleanup bb.
define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %alloc.var = alloca i64
  call void @llvm.lifetime.start.p0(ptr %alloc.var)
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)

  switch i8 %suspend, label %exit [
    i8 0, label %await.suspend
    i8 1, label %exit
  ]
await.suspend:
  %save2 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.handle(ptr null, ptr null, ptr @await_suspend_function)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %await.ready
    i8 1, label %cleanup
  ]
await.ready:
  call void @consume(ptr %alloc.var)
  call void @llvm.lifetime.end.p0(ptr %alloc.var)
  br label %exit

cleanup:
  %free.handle = call ptr @llvm.coro.free(token %id, ptr %vFrame)
  %.not = icmp eq ptr %free.handle, null
  br i1 %.not, label %exit, label %coro.free

coro.free:
  call void @delete(ptr nonnull %free.handle) #2
  br label %exit

exit:
  call void @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; Verify that in the resume part resume call is marked with musttail.
; CHECK-LABEL: @f.resume(
; CHECK:         %[[FRAME:[0-9]+]] = call ptr @await_suspend_function(ptr null, ptr null)
; CHECK:         %[[RESUMEADDR:[0-9]+]] = call ptr @llvm.coro.subfn.addr(ptr %[[FRAME]], i8 0)
; CHECK:         musttail call fastcc void %[[RESUMEADDR]](ptr %[[FRAME]])
; CHECK-NEXT:    ret void

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1
declare i1 @llvm.coro.alloc(token) #2
declare i64 @llvm.coro.size.i64() #3
declare ptr @llvm.coro.begin(token, ptr writeonly) #2
declare token @llvm.coro.save(ptr) #2
declare ptr @llvm.coro.frame() #3
declare i8 @llvm.coro.suspend(token, i1) #2
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #1
declare void @llvm.coro.end(ptr, i1, token) #2
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #1
declare ptr @malloc(i64)
declare void @delete(ptr nonnull) #2
declare void @consume(ptr)
declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
declare ptr @await_suspend_function(ptr %awaiter, ptr %hdl)

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
