; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s
%promise_type = type { i8 }
%struct.A = type <{ i64, i64, i32, [4 x i8] }>

; Function Attrs: noinline ssp uwtable mustprogress
define ptr @foo(ptr nocapture readonly byval(%struct.A) align 8 %a1) #0 {
entry:
  %__promise = alloca %promise_type, align 1
  %a2 = alloca %struct.A, align 8
  %0 = call token @llvm.coro.id(i32 16, ptr nonnull %__promise, ptr @foo, ptr null)
  %1 = call i1 @llvm.coro.alloc(token %0)
  br i1 %1, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
  %2 = call i64 @llvm.coro.size.i64()
  %call = call noalias nonnull ptr @_Znwm(i64 %2) #9
  br label %coro.init

coro.init:                                        ; preds = %coro.alloc, %entry
  %3 = phi ptr [ null, %entry ], [ %call, %coro.alloc ]
  %4 = call ptr @llvm.coro.begin(token %0, ptr %3) #10
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %__promise) #2
  %call2 = call ptr @_ZN4task12promise_type17get_return_objectEv(ptr nonnull dereferenceable(1) %__promise)
  call void @initial_suspend(ptr nonnull dereferenceable(1) %__promise)
  %5 = call token @llvm.coro.save(ptr null)
  call fastcc void @_ZNSt12experimental13coroutines_v116coroutine_handleIN4task12promise_typeEE12from_addressEPv(ptr %4) #2
  %6 = call i8 @llvm.coro.suspend(token %5, i1 false)
  switch i8 %6, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %cleanup33
  ]

init.ready:                                       ; preds = %coro.init
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %a2) #2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a2, ptr align 8 %a1, i64 24, i1 false)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %a2) #2
  call void @_ZN4task12promise_type13final_suspendEv(ptr nonnull dereferenceable(1) %__promise) #2
  %7 = call token @llvm.coro.save(ptr null)
  call fastcc void @_ZNSt12experimental13coroutines_v116coroutine_handleIN4task12promise_typeEE12from_addressEPv(ptr %4) #2
  %8 = call i8 @llvm.coro.suspend(token %7, i1 true) #10
  %switch = icmp ult i8 %8, 2
  br i1 %switch, label %cleanup33, label %coro.ret

cleanup33:                                        ; preds = %init.ready, %coro.init
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %__promise) #2
  %9 = call ptr @llvm.coro.free(token %0, ptr %4)
  %.not = icmp eq ptr %9, null
  br i1 %.not, label %coro.ret, label %coro.free

coro.free:                                        ; preds = %cleanup33
  call void @_ZdlPv(ptr nonnull %9) #2
  br label %coro.ret

coro.ret:                                         ; preds = %coro.free, %cleanup33, %init.ready, %coro.init
  %10 = call i1 @llvm.coro.end(ptr null, i1 false, token none) #10
  ret ptr %call2
}

; check that the frame contains the entire struct, instead of just the struct pointer
; CHECK: %foo.Frame = type { ptr, ptr, %promise_type, %struct.A, i1 }

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nobuiltin nofree allocsize(0)
declare nonnull ptr @_Znwm(i64) local_unnamed_addr #3

; Function Attrs: nounwind readnone
declare i64 @llvm.coro.size.i64() #4

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #6

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare ptr @_ZN4task12promise_type17get_return_objectEv(ptr nonnull dereferenceable(1)) local_unnamed_addr #7 align 2

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare void @initial_suspend(ptr nonnull dereferenceable(1)) local_unnamed_addr #7 align 2

; Function Attrs: nounwind
declare token @llvm.coro.save(ptr) #2

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare hidden fastcc void @_ZNSt12experimental13coroutines_v116coroutine_handleIN4task12promise_typeEE12from_addressEPv(ptr) unnamed_addr #7 align 2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare void @_ZN4task12promise_type13final_suspendEv(ptr nonnull dereferenceable(1)) local_unnamed_addr #7 align 2

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1, token) #2

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(ptr) local_unnamed_addr #8

; Function Attrs: argmemonly nounwind readonly
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #1

attributes #0 = { noinline ssp uwtable mustprogress presplitcoroutine "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nobuiltin nofree allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind readnone }
attributes #5 = { argmemonly nofree nosync nounwind willreturn }
attributes #6 = { argmemonly nofree nounwind willreturn }
attributes #7 = { noinline nounwind ssp uwtable willreturn mustprogress "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #8 = { nobuiltin nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #9 = { allocsize(0) }
attributes #10 = { noduplicate }

