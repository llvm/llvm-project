; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/coro-presplit-massage-III.prof -sample-profile-coro-massage -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/coro-presplit-massage-III.prof -o %t.extbinary.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.extbinary.afdo -sample-profile-coro-massage -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary --use-md5 %S/Inputs/coro-presplit-massage-III.prof -o %t.md5.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.md5.afdo -sample-profile-coro-massage -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/coro-presplit-massage-III.prof -S | FileCheck %s --check-prefix=OFF

; Test 3: End-to-end loop coroutine with .__uniq. suffix and only .resume
; samples in the profile.
; CHECK-LABEL: define internal ptr @_Z9coro_loopv.__uniq.12345()
; CHECK-SAME: !prof ![[ENTRY_PROF:[0-9]+]]
; CHECK: br i1 %r0, label %init.ready, label %init.suspend, {{.*}}!prof ![[INIT_PATH:[0-9]+]]
; CHECK: br i1 %r1, label %sync1.ready, label %sync1.suspend, {{.*}}!prof ![[FAST_PATH:[0-9]+]]
; CHECK: br i1 %r2, label %await2.ready, label %await2.suspend, {{.*}}!prof ![[SUSP_PATH:[0-9]+]]
; CHECK: br i1 %r3, label %sync2.ready, label %sync2.suspend, {{.*}}!prof ![[FAST_PATH]]

; CHECK-DAG: ![[ENTRY_PROF]] = !{!"function_entry_count", i64 1}
; CHECK-DAG: ![[INIT_PATH]] = !{!"branch_weights", i32 2, i32 0}
; CHECK-DAG: ![[FAST_PATH]] = !{!"branch_weights", i32 1, i32 0}
; CHECK-DAG: ![[SUSP_PATH]] = !{!"branch_weights", i32 1, i32 514}

; OFF-LABEL: define internal ptr @_Z9coro_loopv.__uniq.12345()
; OFF-SAME: !prof ![[OFF_ENTRY_PROF:[0-9]+]]
; OFF-NOT: !prof
; OFF: ![[OFF_ENTRY_PROF]] = !{!"function_entry_count", i64 -1}

%struct.promise = type { i64 }
%struct.awaitable = type { i64 }

declare ptr @_Z10coro_allocm(i64)
declare void @_Z9coro_freePv(ptr)
declare i1 @_Z11init_ready_P16struct_awaitable(ptr)
declare void @_Z11init_resumeP16struct_awaitable(ptr)
declare i1 @_Z16sync_await_readyP9awaitable(ptr)
declare i32 @_Z17sync_await_resumeP9awaitable(ptr)
declare i1 @_Z17async_await_readyP9awaitable(ptr)
declare i32 @_Z18async_await_resumeP9awaitable(ptr)
declare void @_Z4workPi(ptr)

declare void @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__init(ptr, ptr) #1
declare void @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__sync1(ptr, ptr) #1
declare void @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__async_loop(ptr, ptr) #1
declare void @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__sync2(ptr, ptr) #1
declare ptr @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__final(ptr, ptr) #1

define internal ptr @_Z9coro_loopv.__uniq.12345() #0 !dbg !6 {
entry:
  %promise = alloca %struct.promise, align 8
  %aw_init = alloca %struct.awaitable, align 8
  %aw_sync1 = alloca %struct.awaitable, align 8
  %aw_async = alloca %struct.awaitable, align 8
  %aw_sync2 = alloca %struct.awaitable, align 8
  %aw_final = alloca %struct.awaitable, align 8
  %slot = alloca i32, align 4
  %id = call token @llvm.coro.id(i32 8, ptr %promise, ptr @_Z9coro_loopv.__uniq.12345, ptr null), !dbg !10
  %need_alloc = call i1 @llvm.coro.alloc(token %id), !dbg !10
  br i1 %need_alloc, label %coro.alloc, label %coro.init, !dbg !10

coro.alloc:
  %sz = call i64 @llvm.coro.size.i64(), !dbg !10
  %mem = call ptr @_Z10coro_allocm(i64 %sz), !dbg !11
  br label %coro.init, !dbg !11

coro.init:
  %phi_mem = phi ptr [ null, %entry ], [ %mem, %coro.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi_mem), !dbg !10
  %r0 = call i1 @_Z11init_ready_P16struct_awaitable(ptr %aw_init), !dbg !12
  br i1 %r0, label %init.ready, label %init.suspend, !dbg !12

init.suspend:
  %s0 = call token @llvm.coro.save(ptr null), !dbg !10
  call void @llvm.coro.await.suspend.void(ptr %aw_init, ptr %hdl, ptr @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__init), !dbg !10
  %sp0 = call i8 @llvm.coro.suspend(token %s0, i1 false), !dbg !10
  switch i8 %sp0, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %cleanup0
  ], !dbg !13

init.ready:
  call void @_Z11init_resumeP16struct_awaitable(ptr %aw_init), !dbg !14
  br label %cleanup0, !dbg !14

cleanup0:
  %dest0 = phi i32 [ 0, %init.ready ], [ 2, %init.suspend ]
  switch i32 %dest0, label %unreachable [
    i32 0, label %pre.sync1
    i32 2, label %coro.cleanup
  ]

pre.sync1:
  %r1 = call i1 @_Z16sync_await_readyP9awaitable(ptr %aw_sync1), !dbg !15
  br i1 %r1, label %sync1.ready, label %sync1.suspend, !dbg !15

sync1.suspend:
  %s1 = call token @llvm.coro.save(ptr null), !dbg !15
  call void @llvm.coro.await.suspend.void(ptr %aw_sync1, ptr %hdl, ptr @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__sync1), !dbg !15
  %sp1 = call i8 @llvm.coro.suspend(token %s1, i1 false), !dbg !15
  switch i8 %sp1, label %coro.ret [
    i8 0, label %sync1.ready
    i8 1, label %cleanup1
  ], !dbg !16

sync1.ready:
  %v1 = call i32 @_Z17sync_await_resumeP9awaitable(ptr %aw_sync1), !dbg !17
  br label %cleanup1, !dbg !17

cleanup1:
  %dest1 = phi i32 [ 0, %sync1.ready ], [ 2, %sync1.suspend ]
  %c1 = icmp eq i32 %dest1, 0
  br i1 %c1, label %for.preheader, label %cleanup.dispatch

for.preheader:
  br label %for.cond, !dbg !18

for.cond:
  %i = phi i32 [ 0, %for.preheader ], [ %i.next, %for.inc ]
  %cmp = icmp slt i32 %i, 500, !dbg !19
  br i1 %cmp, label %for.body, label %loop.exit, !dbg !19

for.body:
  %r2 = call i1 @_Z17async_await_readyP9awaitable(ptr %aw_async), !dbg !20
  br i1 %r2, label %await2.ready, label %await2.suspend, !dbg !20

await2.suspend:
  %s2 = call token @llvm.coro.save(ptr null), !dbg !20
  call void @llvm.coro.await.suspend.void(ptr %aw_async, ptr %hdl, ptr @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__async_loop), !dbg !20
  %sp2 = call i8 @llvm.coro.suspend(token %s2, i1 false), !dbg !20
  switch i8 %sp2, label %coro.ret [
    i8 0, label %await2.ready
    i8 1, label %cleanup2
  ], !dbg !21

await2.ready:
  %v2 = call i32 @_Z18async_await_resumeP9awaitable(ptr %aw_async), !dbg !22
  store i32 %v2, ptr %slot, align 4, !dbg !22
  br label %cleanup2, !dbg !22

cleanup2:
  %dest2 = phi i32 [ 0, %await2.ready ], [ 2, %await2.suspend ]
  %c2 = icmp eq i32 %dest2, 0
  br i1 %c2, label %after.await2, label %cleanup.loop

after.await2:
  %v2.reload = load volatile i32, ptr %slot, align 4, !dbg !23
  br label %inner.cond, !dbg !23

inner.cond:
  %k = phi i32 [ 0, %after.await2 ], [ %k.next, %inner.body ]
  %kcmp = icmp slt i32 %k, 10, !dbg !24
  br i1 %kcmp, label %inner.body, label %inner.end, !dbg !24

inner.body:
  call void @_Z4workPi(ptr %slot), !dbg !25
  %k.next = add nsw i32 %k, 1, !dbg !24
  br label %inner.cond, !dbg !24

inner.end:
  br label %cleanup.loop, !dbg !26

cleanup.loop:
  %dest3 = phi i32 [ %dest2, %cleanup2 ], [ 0, %inner.end ]
  %c3 = icmp eq i32 %dest3, 0
  br i1 %c3, label %for.inc, label %loop.exit

for.inc:
  %i.next = add nsw i32 %i, 1, !dbg !27
  br label %for.cond, !dbg !27

loop.exit:
  %dest4 = phi i32 [ %dest3, %cleanup.loop ], [ 4, %for.cond ]
  %c4 = icmp eq i32 %dest4, 4
  br i1 %c4, label %post.loop, label %cleanup.dispatch

post.loop:
  %r3 = call i1 @_Z16sync_await_readyP9awaitable(ptr %aw_sync2), !dbg !28
  br i1 %r3, label %sync2.ready, label %sync2.suspend, !dbg !28

sync2.suspend:
  %s3 = call token @llvm.coro.save(ptr null), !dbg !28
  call void @llvm.coro.await.suspend.void(ptr %aw_sync2, ptr %hdl, ptr @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__sync2), !dbg !28
  %sp3 = call i8 @llvm.coro.suspend(token %s3, i1 false), !dbg !28
  switch i8 %sp3, label %coro.ret [
    i8 0, label %sync2.ready
    i8 1, label %cleanup3
  ], !dbg !29

sync2.ready:
  %v3 = call i32 @_Z17sync_await_resumeP9awaitable(ptr %aw_sync2), !dbg !30
  br label %cleanup3, !dbg !30

cleanup3:
  %dest5 = phi i32 [ 0, %sync2.ready ], [ 2, %sync2.suspend ]
  %c5 = icmp eq i32 %dest5, 0
  br i1 %c5, label %before.final, label %cleanup.dispatch

before.final:
  br label %cleanup.dispatch

cleanup.dispatch:
  %dest6 = phi i32 [ %dest1, %cleanup1 ], [ %dest4, %loop.exit ], [ %dest5, %cleanup3 ], [ 3, %before.final ]
  switch i32 %dest6, label %unreachable [
    i32 2, label %coro.cleanup
    i32 3, label %coro.final
  ]

coro.final:
  %sf = call token @llvm.coro.save(ptr null), !dbg !10
  call void @llvm.coro.await.suspend.handle(ptr %aw_final, ptr %hdl, ptr @_Z9coro_loopv.__uniq.12345.__await_suspend_wrapper__final), !dbg !10
  %spf = call i8 @llvm.coro.suspend(token %sf, i1 true), !dbg !10
  switch i8 %spf, label %coro.ret [
    i8 0, label %final.ready
    i8 1, label %coro.cleanup
  ], !dbg !31

final.ready:
  br label %coro.cleanup

coro.cleanup:
  %free_ptr = call ptr @llvm.coro.free(token %id, ptr %hdl), !dbg !10
  call void @_Z9coro_freePv(ptr %free_ptr), !dbg !10
  br label %coro.ret

coro.ret:
  call void @llvm.coro.end(ptr null, i1 false, token none), !dbg !10
  ret ptr %hdl, !dbg !10

unreachable:
  unreachable
}

declare token @llvm.coro.id(i32, ptr readnone, ptr readonly, ptr)
declare i1 @llvm.coro.alloc(token)
declare i64 @llvm.coro.size.i64()
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare token @llvm.coro.save(ptr)
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare void @llvm.coro.await.suspend.handle(ptr, ptr, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr readonly)
declare void @llvm.coro.end(ptr, i1, token)

attributes #0 = { presplitcoroutine "use-sample-profile" }
attributes #1 = { "sample-profile-suffix-elision-policy"="selected" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, debugInfoForProfiling: true)
!1 = !DIFile(filename: "coro.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !2)
!6 = distinct !DISubprogram(name: "coro_loop", linkageName: "_Z9coro_loopv.__uniq.12345", scope: !1, file: !1, line: 100, type: !5, scopeLine: 100, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 2)
!8 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 4)
!9 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 6)
!10 = !DILocation(line: 100, column: 1, scope: !6)
!11 = !DILocation(line: 100, column: 5, scope: !7)
!12 = !DILocation(line: 100, column: 9, scope: !8)
!13 = !DILocation(line: 100, column: 9, scope: !7)
!14 = !DILocation(line: 100, column: 9, scope: !32)
!15 = !DILocation(line: 102, column: 3, scope: !6)
!16 = !DILocation(line: 102, column: 3, scope: !7)
!17 = !DILocation(line: 102, column: 3, scope: !9)
!18 = !DILocation(line: 103, column: 3, scope: !6)
!19 = !DILocation(line: 103, column: 15, scope: !7)
!20 = !DILocation(line: 104, column: 5, scope: !6)
!21 = !DILocation(line: 104, column: 5, scope: !7)
!22 = !DILocation(line: 104, column: 5, scope: !9)
!23 = !DILocation(line: 105, column: 5, scope: !6)
!24 = !DILocation(line: 106, column: 7, scope: !7)
!25 = !DILocation(line: 107, column: 9, scope: !6)
!26 = !DILocation(line: 108, column: 5, scope: !6)
!27 = !DILocation(line: 108, column: 10, scope: !9)
!28 = !DILocation(line: 110, column: 3, scope: !6)
!29 = !DILocation(line: 110, column: 3, scope: !7)
!30 = !DILocation(line: 110, column: 3, scope: !9)
!31 = !DILocation(line: 100, column: 15, scope: !8)
!32 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 12)
