; RUN: opt < %s -passes='cgscc(coro-split),early-cse,dce,simplifycfg' -S | FileCheck %s

%"struct.std::__n4861::noop_coroutine_promise" = type { i8 }
%struct.Promise = type { %"struct.std::__n4861::coroutine_handle" }
%"struct.std::__n4861::coroutine_handle" = type { ptr }

define ptr @foo() #1 {
entry:
  %__promise = alloca %struct.Promise, align 8
  %0 = call token @llvm.coro.id(i32 16, ptr nonnull %__promise, ptr nonnull @foo, ptr null)
  %1 = call i1 @llvm.coro.alloc(token %0)
  br i1 %1, label %coro.alloc, label %init.suspend

coro.alloc:                                       ; preds = %entry
  %2 = tail call i64 @llvm.coro.size.i64()
  %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef %2) #11
  br label %init.suspend

init.suspend:                                     ; preds = %entry, %coro.alloc
  %3 = phi ptr [ null, %entry ], [ %call, %coro.alloc ]
  %4 = call ptr @llvm.coro.begin(token %0, ptr %3) #12
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %__promise) #3
  store ptr null, ptr %__promise, align 8
  %5 = call token @llvm.coro.save(ptr null)
  %6 = call i8 @llvm.coro.suspend(token %5, i1 false)
  switch i8 %6, label %coro.ret [
  i8 0, label %await.suspend
  i8 1, label %cleanup1
  ]

await.suspend:                                    ; preds = %init.suspend
  %7 = call token @llvm.coro.save(ptr null)
  %8 = call i8 @llvm.coro.suspend(token %7, i1 false)
  switch i8 %8, label %coro.ret [
  i8 0, label %await2.suspend
  i8 1, label %cleanup2
  ]

await2.suspend:                                   ; preds = %await.suspend
  %call27 = call ptr @_Z5Innerv() #3
  %9 = call token @llvm.coro.save(ptr null)
  %10 = getelementptr inbounds i8, ptr %__promise, i64 -16
  store ptr %10, ptr %call27, align 8
  %11 = getelementptr inbounds i8, ptr %call27, i64 -16
  %12 = call ptr @llvm.coro.subfn.addr(ptr nonnull %11, i8 0)
  call fastcc void %12(ptr nonnull %11) #3
  %13 = call i8 @llvm.coro.suspend(token %9, i1 false)
  switch i8 %13, label %coro.ret [
  i8 0, label %final.suspend
  i8 1, label %cleanup3
  ]

final.suspend:                                    ; preds = %await2.suspend
  %14 = call ptr @llvm.coro.subfn.addr(ptr nonnull %11, i8 1)
  call fastcc void %14(ptr nonnull %11) #3
  %15 = call token @llvm.coro.save(ptr null)
  %retval.sroa.0.0.copyload.i = load ptr, ptr %__promise, align 8
  %16 = call ptr @llvm.coro.subfn.addr(ptr %retval.sroa.0.0.copyload.i, i8 0)
  call fastcc void %16(ptr %retval.sroa.0.0.copyload.i) #3
  %17 = call i8 @llvm.coro.suspend(token %15, i1 true) #12
  switch i8 %17, label %coro.ret [
  i8 0, label %final.ready
  i8 1, label %cleanup62
  ]

final.ready:                                      ; preds = %final.suspend
  call void @exit(i32 noundef 1)
  unreachable

cleanup1:
  call void @dtor1()
  br label %cleanup62

cleanup2:
  call void @dtor2()
  br label %cleanup62

cleanup3:
  call void @dtor3()
  br label %cleanup62

cleanup62:                                        ; preds = %await2.suspend, %await.suspend, %init.suspend, %final.suspend
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %__promise) #3
  %18 = call ptr @llvm.coro.free(token %0, ptr %4)
  %.not = icmp eq ptr %18, null
  br i1 %.not, label %coro.ret, label %coro.free

coro.free:                                        ; preds = %cleanup62
  call void @_ZdlPv(ptr noundef nonnull %18) #3
  br label %coro.ret

coro.ret:                                         ; preds = %coro.free, %cleanup62, %final.suspend, %await2.suspend, %await.suspend, %init.suspend
  %19 = call i1 @llvm.coro.end(ptr null, i1 false, token none) #12
  ret ptr %__promise
}

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #2
declare i1 @llvm.coro.alloc(token) #3
declare dso_local noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #4
declare i64 @llvm.coro.size.i64() #5
declare ptr @llvm.coro.begin(token, ptr writeonly) #3
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #6
declare token @llvm.coro.save(ptr) #7
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #6
declare i8 @llvm.coro.suspend(token, i1) #3
declare ptr @_Z5Innerv() local_unnamed_addr
declare dso_local void @_ZdlPv(ptr noundef) local_unnamed_addr #8
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2
declare i1 @llvm.coro.end(ptr, i1, token) #3
declare void @exit(i32 noundef)
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #10
declare void @dtor1()
declare void @dtor2()
declare void @dtor3()

attributes #0 = { mustprogress nounwind uwtable }
attributes #1 = { nounwind presplitcoroutine uwtable coro_only_destroy_when_complete }
attributes #2 = { argmemonly nofree nounwind readonly }
attributes #3 = { nounwind }
attributes #4 = { nobuiltin allocsize(0) }
attributes #5 = { nofree nosync nounwind readnone }
attributes #6 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #7 = { nomerge nounwind }
attributes #8 = { nobuiltin nounwind }
attributes #9 = { noreturn }
attributes #10 = { argmemonly nounwind readonly }
attributes #11 = { nounwind allocsize(0) }
attributes #12 = { noduplicate }

; CHECK: define{{.*}}@foo.destroy(
; CHECK-NEXT: entry.destroy:
; CHECK-NEXT: call void @_ZdlPv
; CHECK-NEXT: ret void

; CHECK: define{{.*}}@foo.cleanup(
; CHECK-NEXT: entry.cleanup:
; CHECK-NEXT: ret void
