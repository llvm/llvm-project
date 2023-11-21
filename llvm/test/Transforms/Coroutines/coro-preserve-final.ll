; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%"struct.std::__n4861::noop_coroutine_promise" = type { i8 }
%struct.Promise = type { %"struct.std::__n4861::coroutine_handle" }
%"struct.std::__n4861::coroutine_handle" = type { ptr }

define dso_local ptr @_Z5Outerv() #1 {
entry:
  %__promise = alloca %struct.Promise, align 8
  %0 = call token @llvm.coro.id(i32 16, ptr nonnull %__promise, ptr nonnull @_Z5Outerv, ptr null)
  %1 = call i1 @llvm.coro.alloc(token %0)
  br i1 %1, label %coro.alloc, label %init.suspend

coro.alloc:                                       ; preds = %entry
  %2 = tail call i64 @llvm.coro.size.i64()
  %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef %2) #12
  br label %init.suspend

init.suspend:                                     ; preds = %entry, %coro.alloc
  %3 = phi ptr [ null, %entry ], [ %call, %coro.alloc ]
  %4 = call ptr @llvm.coro.begin(token %0, ptr %3) #13
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %__promise) #3
  store ptr null, ptr %__promise, align 8
  %5 = call token @llvm.coro.save(ptr null)
  %6 = call i8 @llvm.coro.suspend(token %5, i1 false)
  switch i8 %6, label %coro.ret [
  i8 0, label %await.suspend
  i8 1, label %cleanup62
  ]

await.suspend:                                    ; preds = %init.suspend
  %7 = call token @llvm.coro.save(ptr null)
  %8 = call ptr @llvm.coro.subfn.addr(ptr %4, i8 0)
  call fastcc void %8(ptr %4) #3
  %9 = call i8 @llvm.coro.suspend(token %7, i1 false)
  switch i8 %9, label %coro.ret [
  i8 0, label %await2.suspend
  i8 1, label %cleanup62
  ]

await2.suspend:                                   ; preds = %await.suspend
  %call27 = call ptr @_Z5Innerv() #3
  %10 = call token @llvm.coro.save(ptr null)
  %11 = getelementptr inbounds i8, ptr %__promise, i64 -16
  store ptr %11, ptr %call27, align 8
  %12 = getelementptr inbounds i8, ptr %call27, i64 -16
  %13 = call ptr @llvm.coro.subfn.addr(ptr nonnull %12, i8 0)
  call fastcc void %13(ptr nonnull %12) #3
  %14 = call i8 @llvm.coro.suspend(token %10, i1 false)
  switch i8 %14, label %coro.ret [
  i8 0, label %final.suspend
  i8 1, label %cleanup62
  ]

final.suspend:                                    ; preds = %await2.suspend
  %15 = call ptr @llvm.coro.subfn.addr(ptr nonnull %12, i8 1)
  call fastcc void %15(ptr nonnull %12) #3
  %16 = call token @llvm.coro.save(ptr null)
  %retval.sroa.0.0.copyload.i = load ptr, ptr %__promise, align 8
  %17 = call ptr @llvm.coro.subfn.addr(ptr %retval.sroa.0.0.copyload.i, i8 0)
  call fastcc void %17(ptr %retval.sroa.0.0.copyload.i) #3
  %18 = call i8 @llvm.coro.suspend(token %16, i1 true) #13
  switch i8 %18, label %coro.ret [
  i8 0, label %final.ready
  i8 1, label %cleanup62
  ]

final.ready:                                      ; preds = %final.suspend
  call void @_Z5_exiti(i32 noundef 1) #14
  unreachable

cleanup62:                                        ; preds = %await2.suspend, %await.suspend, %init.suspend, %final.suspend
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %__promise) #3
  %19 = call ptr @llvm.coro.free(token %0, ptr %4)
  %.not = icmp eq ptr %19, null
  br i1 %.not, label %coro.ret, label %coro.free

coro.free:                                        ; preds = %cleanup62
  call void @_ZdlPv(ptr noundef nonnull %19) #3
  br label %coro.ret

coro.ret:                                         ; preds = %coro.free, %cleanup62, %final.suspend, %await2.suspend, %await.suspend, %init.suspend
  %20 = call i1 @llvm.coro.end(ptr null, i1 false, token none) #13
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
declare dso_local ptr @_Z5Innerv() local_unnamed_addr #8
declare dso_local void @_ZdlPv(ptr noundef) local_unnamed_addr #9
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2
declare i1 @llvm.coro.end(ptr, i1, token) #3
declare dso_local void @_Z5_exiti(i32 noundef) local_unnamed_addr #10
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #11

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind presplitcoroutine uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { argmemonly nofree nounwind readonly }
attributes #3 = { nounwind }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nofree nosync nounwind readnone }
attributes #6 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #7 = { nomerge nounwind }
attributes #8 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { noreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { argmemonly nounwind readonly }
attributes #12 = { nounwind allocsize(0) }
attributes #13 = { noduplicate }
attributes #14 = { noreturn nounwind }

; CHECK: define{{.*}}@_Z5Outerv.resume(
; CHECK: entry.resume:
; CHECK: switch i2 %index
; CHECK-NEXT:    i2 0, label %await2.suspend
; CHECK-NEXT:    i2 1, label %final.suspend
;
; CHECK: await2.suspend:
; CHECK: musttail call
; CHECK-NEXT: ret void
;
; CHECK: final.suspend:
; CHECK: musttail call
; CHECK-NEXT: ret void
