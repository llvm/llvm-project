; Tests that coro-split will convert coro.resume followed by a suspend to a
; musttail call.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck --check-prefixes=CHECK,NOPGO %s
; RUN: opt < %s -passes='pgo-instr-gen,cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck --check-prefixes=CHECK,PGO %s

define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %addr1 = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
  call fastcc void %addr1(ptr null)

  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  %cmp = icmp eq i8 %suspend, 0
  br i1 %cmp, label %await.suspend, label %exit
await.suspend:
  %save2 = call token @llvm.coro.save(ptr null)
  %br0 = call i8 @switch_result()
  switch i8 %br0, label %unreach [
    i8 0, label %await.resume3
    i8 1, label %await.resume1
    i8 2, label %await.resume2
  ]
await.resume1:
  %hdl = call ptr @g()
  %addr2 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %addr2(ptr %hdl)
  br label %final.suspend
await.resume2:
  %hdl2 = call ptr @h()
  %addr3 = call ptr @llvm.coro.subfn.addr(ptr %hdl2, i8 0)
  call fastcc void %addr3(ptr %hdl2)
  br label %final.suspend
await.resume3:
  %addr4 = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
  call fastcc void %addr4(ptr null)
  br label %final.suspend
final.suspend:
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  %cmp2 = icmp eq i8 %suspend2, 0
  br i1 %cmp2, label %pre.exit, label %exit
pre.exit:
  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
unreach:
  unreachable
}

; Verify that in the initial function resume is not marked with musttail.
; CHECK-LABEL: @f(
; CHECK: %[[addr1:.+]] = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
; CHECK-NOT: musttail call fastcc void %[[addr1]](ptr null)

; Verify that in the resume part resume call is marked with musttail.
; CHECK-LABEL: @f.resume(
; CHECK: %[[hdl:.+]] = call ptr @g()
; CHECK-NEXT: %[[addr2:.+]] = call ptr @llvm.coro.subfn.addr(ptr %[[hdl]], i8 0)
; NOPGO-NEXT: musttail call fastcc void %[[addr2]](ptr %[[hdl]])
; PGO: musttail call fastcc void %[[addr2]](ptr %[[hdl]])
; CHECK-NEXT: ret void
; CHECK: %[[hdl2:.+]] = call ptr @h()
; CHECK-NEXT: %[[addr3:.+]] = call ptr @llvm.coro.subfn.addr(ptr %[[hdl2]], i8 0)
; NOPGO-NEXT: musttail call fastcc void %[[addr3]](ptr %[[hdl2]])
; PGO: musttail call fastcc void %[[addr3]](ptr %[[hdl2]])
; CHECK-NEXT: ret void
; CHECK: %[[addr4:.+]] = call ptr @llvm.coro.subfn.addr(ptr null, i8 0)
; NOPGO-NEXT: musttail call fastcc void %[[addr4]](ptr null)
; PGO: musttail call fastcc void %[[addr4]](ptr null)
; CHECK-NEXT: ret void



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
declare i8 @switch_result()
declare ptr @g()
declare ptr @h()

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
