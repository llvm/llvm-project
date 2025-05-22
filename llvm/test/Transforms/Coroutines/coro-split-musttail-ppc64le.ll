; Tests that some target (e.g. ppc) can support tail call under condition.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S \
; RUN:     -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 | FileCheck %s
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S \
; RUN:     -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr10 --code-model=medium \
; RUN:     | FileCheck %s
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S \
; RUN:     -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr10 --code-model=medium -mattr=+longcall \
; RUN:     | FileCheck %s
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S \
; RUN:     -mtriple=ppc32-- | FileCheck %s

define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %addr1 = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
  call fastcc void %addr1(ptr null)

  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %save2 = call token @llvm.coro.save(ptr null)
  %addr2 = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
  call fastcc void %addr2(ptr null)

  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %exit
    i8 1, label %exit
  ]
exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; Verify that in the initial function resume is not marked with musttail.
; CHECK-LABEL: @f(
; CHECK: %[[addr1:.+]] = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
; CHECK-NOT: musttail call fastcc void %[[addr1]](ptr null)

; Verify that ppc target not using PC-Relative addressing in the resume part resume call is not marked with musttail.
; CHECK-LABEL: @f.resume(
; CHECK: %[[addr2:.+]] = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
; CHECK-NEXT: call fastcc void %[[addr2]](ptr null)

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1
declare i1 @llvm.coro.alloc(token) #2
declare i64 @llvm.coro.size.i64() #3
declare ptr @llvm.coro.begin(token, ptr writeonly) #2
declare token @llvm.coro.save(ptr) #2
declare ptr @llvm.coro.frame() #3
declare i8 @llvm.coro.suspend(token, i1) #2
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #1
declare i1 @llvm.coro.end(ptr, i1, token) #2
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #1
declare ptr @malloc(i64)

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
