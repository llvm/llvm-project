; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s
%promise_type = type { i8 }
%struct.A = type <{ i64, i64, i32, [4 x i8] }>

; Function Attrs: noinline ssp uwtable mustprogress
define %promise_type* @foo(%struct.A* nocapture readonly byval(%struct.A) align 8 %a1) #0 {
entry:
  %__promise = alloca %promise_type, align 1
  %a2 = alloca %struct.A, align 8
  %0 = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 0
  %1 = call token @llvm.coro.id(i32 16, i8* nonnull %0, i8* bitcast (%promise_type* (%struct.A*)* @foo to i8*), i8* null)
  %2 = call i1 @llvm.coro.alloc(token %1)
  br i1 %2, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
  %3 = call i64 @llvm.coro.size.i64()
  %call = call noalias nonnull i8* @_Znwm(i64 %3) #9
  br label %coro.init

coro.init:                                        ; preds = %coro.alloc, %entry
  %4 = phi i8* [ null, %entry ], [ %call, %coro.alloc ]
  %5 = call i8* @llvm.coro.begin(token %1, i8* %4) #10
  %6 = bitcast %struct.A* %a1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0) #2
  %call2 = call %promise_type* @_ZN4task12promise_type17get_return_objectEv(%promise_type* nonnull dereferenceable(1) %__promise)
  call void @initial_suspend(%promise_type* nonnull dereferenceable(1) %__promise)
  %7 = call token @llvm.coro.save(i8* null)
  call fastcc void @_ZNSt12experimental13coroutines_v116coroutine_handleIN4task12promise_typeEE12from_addressEPv(i8* %5) #2
  %8 = call i8 @llvm.coro.suspend(token %7, i1 false)
  switch i8 %8, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %cleanup33
  ]

init.ready:                                       ; preds = %coro.init
  %9 = bitcast %struct.A* %a2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %9) #2
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 8 %6, i64 24, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %9) #2
  call void @_ZN4task12promise_type13final_suspendEv(%promise_type* nonnull dereferenceable(1) %__promise) #2
  %10 = call token @llvm.coro.save(i8* null)
  call fastcc void @_ZNSt12experimental13coroutines_v116coroutine_handleIN4task12promise_typeEE12from_addressEPv(i8* %5) #2
  %11 = call i8 @llvm.coro.suspend(token %10, i1 true) #10
  %switch = icmp ult i8 %11, 2
  br i1 %switch, label %cleanup33, label %coro.ret

cleanup33:                                        ; preds = %init.ready, %coro.init
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0) #2
  %12 = call i8* @llvm.coro.free(token %1, i8* %5)
  %.not = icmp eq i8* %12, null
  br i1 %.not, label %coro.ret, label %coro.free

coro.free:                                        ; preds = %cleanup33
  call void @_ZdlPv(i8* nonnull %12) #2
  br label %coro.ret

coro.ret:                                         ; preds = %coro.free, %cleanup33, %init.ready, %coro.init
  %13 = call i1 @llvm.coro.end(i8* null, i1 false) #10
  ret %promise_type* %call2
}

; check that the frame contains the entire struct, instead of just the struct pointer
; CHECK: %foo.Frame = type { void (%foo.Frame*)*, void (%foo.Frame*)*, %promise_type, %struct.A, i1 }

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nobuiltin nofree allocsize(0)
declare nonnull i8* @_Znwm(i64) local_unnamed_addr #3

; Function Attrs: nounwind readnone
declare i64 @llvm.coro.size.i64() #4

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #6

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare %promise_type* @_ZN4task12promise_type17get_return_objectEv(%promise_type* nonnull dereferenceable(1)) local_unnamed_addr #7 align 2

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare void @initial_suspend(%promise_type* nonnull dereferenceable(1)) local_unnamed_addr #7 align 2

; Function Attrs: nounwind
declare token @llvm.coro.save(i8*) #2

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare hidden fastcc void @_ZNSt12experimental13coroutines_v116coroutine_handleIN4task12promise_typeEE12from_addressEPv(i8*) unnamed_addr #7 align 2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: noinline nounwind ssp uwtable willreturn mustprogress
declare void @_ZN4task12promise_type13final_suspendEv(%promise_type* nonnull dereferenceable(1)) local_unnamed_addr #7 align 2

; Function Attrs: nounwind
declare i1 @llvm.coro.end(i8*, i1) #2

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) local_unnamed_addr #8

; Function Attrs: argmemonly nounwind readonly
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #1

attributes #0 = { noinline ssp uwtable mustprogress "coroutine.presplit"="1" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
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

