; MIPS makes tail calls only with -mips-tail-calls, and never in MIPS16 code,
; so the resume call of a symmetric transfer is marked musttail only then: a
; musttail call that the backend cannot make is a fatal error.
; REQUIRES: mips-registered-target
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S -mtriple=mips-linux-gnu | FileCheck %s --check-prefixes=CHECK,NOTAIL
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S -mtriple=mips64-linux-gnuabi64 | FileCheck %s --check-prefixes=CHECK,NOTAIL
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S -mtriple=mips-linux-gnu -mattr=+mips16 -mips-tail-calls | FileCheck %s --check-prefixes=CHECK,NOTAIL
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S -mtriple=mips-linux-gnu -mips-tail-calls | FileCheck %s --check-prefixes=CHECK,TAIL
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S -mtriple=mips64-linux-gnuabi64 -mips-tail-calls | FileCheck %s --check-prefixes=CHECK,TAIL

define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %addr1 = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
  call void %addr1(ptr null)

  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %save2 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.handle(ptr null, ptr null, ptr @await_suspend_function)

  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %exit
    i8 1, label %exit
  ]
exit:
  call void @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; CHECK-LABEL: @f.resume(
; CHECK: %[[addr2:.+]] = call ptr @llvm.coro.subfn.addr
; NOTAIL-NEXT: {{^  }}call void %[[addr2]]
; TAIL-NEXT: musttail call void %[[addr2]]
; TAIL-NEXT: ret void

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
declare ptr @await_suspend_function(ptr %awaiter, ptr %hdl)

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
