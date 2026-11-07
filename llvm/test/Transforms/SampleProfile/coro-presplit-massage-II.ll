; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/coro-presplit-massage-II.prof -sample-profile-coro-massage -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/coro-presplit-massage-II.prof -o %t.extbinary.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.extbinary.afdo -sample-profile-coro-massage -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary --use-md5 %S/Inputs/coro-presplit-massage-II.prof -o %t.md5.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.md5.afdo -sample-profile-coro-massage -S | FileCheck %s

; Test 2: Fast-path co_await is not inverted by 0-count .resume records, and
; slow-path co_await sums wrapper samples across .ramp (200) and .resume (800).
; CHECK-LABEL: define ptr @_Z16coro_ramp_resumev()
; CHECK-SAME: !prof ![[ENTRY_PROF:[0-9]+]]
; CHECK: br i1 %r1, label %sync.ready, label %sync.suspend, {{.*}}!prof ![[SYNC_BR:[0-9]+]]
; CHECK: br i1 %r2, label %async.ready, label %async.suspend, {{.*}}!prof ![[ASYNC_BR:[0-9]+]]

; CHECK-DAG: ![[ENTRY_PROF]] = !{!"function_entry_count", i64 201}
; CHECK-DAG: ![[SYNC_BR]] = !{!"branch_weights", i32 202, i32 0}
; CHECK-DAG: ![[ASYNC_BR]] = !{!"branch_weights", i32 1, i32 1001}

declare i1 @_Z16sync_await_readyP9awaitable(ptr)
declare i32 @_Z17sync_await_resumeP9awaitable(ptr)
declare i1 @_Z17async_await_readyP9awaitable(ptr)
declare i32 @_Z18async_await_resumeP9awaitable(ptr)
declare void @_Z16coro_ramp_resumev.__await_suspend_wrapper__sync(ptr, ptr) #1
declare void @_Z16coro_ramp_resumev.__await_suspend_wrapper__async(ptr, ptr) #1

define ptr @_Z16coro_ramp_resumev() #0 !dbg !6 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @_Z16coro_ramp_resumev, ptr null), !dbg !7
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null), !dbg !7
  %r1 = call i1 @_Z16sync_await_readyP9awaitable(ptr null), !dbg !8
  br i1 %r1, label %sync.ready, label %sync.suspend, !dbg !8

sync.suspend:
  %s1 = call token @llvm.coro.save(ptr null), !dbg !8
  call void @llvm.coro.await.suspend.void(ptr null, ptr %hdl, ptr @_Z16coro_ramp_resumev.__await_suspend_wrapper__sync), !dbg !8
  %sp1 = call i8 @llvm.coro.suspend(token %s1, i1 false), !dbg !8
  switch i8 %sp1, label %coro.ret [
    i8 0, label %sync.ready
    i8 1, label %coro.ret
  ], !dbg !9

sync.ready:
  %v1 = call i32 @_Z17sync_await_resumeP9awaitable(ptr null), !dbg !10
  br label %after.sync, !dbg !10

after.sync:
  %r2 = call i1 @_Z17async_await_readyP9awaitable(ptr null), !dbg !11
  br i1 %r2, label %async.ready, label %async.suspend, !dbg !11

async.suspend:
  %s2 = call token @llvm.coro.save(ptr null), !dbg !11
  call void @llvm.coro.await.suspend.void(ptr null, ptr %hdl, ptr @_Z16coro_ramp_resumev.__await_suspend_wrapper__async), !dbg !11
  %sp2 = call i8 @llvm.coro.suspend(token %s2, i1 false), !dbg !11
  switch i8 %sp2, label %coro.ret [
    i8 0, label %async.ready
    i8 1, label %coro.ret
  ], !dbg !12

async.ready:
  %v2 = call i32 @_Z18async_await_resumeP9awaitable(ptr null), !dbg !13
  br label %coro.ret, !dbg !13

coro.ret:
  call void @llvm.coro.end(ptr null, i1 false, token none), !dbg !7
  ret ptr %hdl, !dbg !7
}

declare token @llvm.coro.id(i32, ptr readnone, ptr readonly, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare token @llvm.coro.save(ptr)
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare i8 @llvm.coro.suspend(token, i1)
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
!6 = distinct !DISubprogram(name: "coro_ramp_resume", linkageName: "_Z16coro_ramp_resumev", scope: !1, file: !1, line: 20, type: !5, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILocation(line: 20, column: 1, scope: !6)
!8 = !DILocation(line: 21, column: 3, scope: !6)
!9 = !DILocation(line: 21, column: 3, scope: !14)
!10 = !DILocation(line: 21, column: 3, scope: !15)
!11 = !DILocation(line: 23, column: 3, scope: !6)
!12 = !DILocation(line: 23, column: 3, scope: !14)
!13 = !DILocation(line: 23, column: 3, scope: !15)
!14 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 2)
!15 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 6)
