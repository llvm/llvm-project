; Tests that we would convert coro.resume to a musttail call if the target is
; Wasm32 with tail-call support.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

target triple = "wasm32-unknown-unknown"

define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %alloc = call i8* @malloc(i64 16) #3
  %vFrame = call noalias nonnull i8* @llvm.coro.begin(token %id, i8* %alloc)

  %save = call token @llvm.coro.save(i8* null)
  %addr1 = call i8* @llvm.coro.subfn.addr(i8* null, i8 0)
  %pv1 = bitcast i8* %addr1 to void (i8*)*
  call fastcc void %pv1(i8* null)

  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %save2 = call token @llvm.coro.save(i8* null)
  %addr2 = call i8* @llvm.coro.subfn.addr(i8* null, i8 0)
  %pv2 = bitcast i8* %addr2 to void (i8*)*
  call fastcc void %pv2(i8* null)

  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %exit
    i8 1, label %exit
  ]
exit:
  call i1 @llvm.coro.end(i8* null, i1 false)
  ret void
}

; CHECK: musttail call

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*) #1
declare i1 @llvm.coro.alloc(token) #2
declare i64 @llvm.coro.size.i64() #3
declare i8* @llvm.coro.begin(token, i8* writeonly) #2
declare token @llvm.coro.save(i8*) #2
declare i8* @llvm.coro.frame() #3
declare i8 @llvm.coro.suspend(token, i1) #2
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #1
declare i1 @llvm.coro.end(i8*, i1) #2
declare i8* @llvm.coro.subfn.addr(i8* nocapture readonly, i8) #1
declare i8* @malloc(i64)

attributes #0 = { presplitcoroutine "target-features"="+tail-call" }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
