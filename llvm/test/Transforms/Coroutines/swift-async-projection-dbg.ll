; RUN: opt %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s

; The async context projection at the start of a resume funclet updates the
; async context slot of the frame. Debuggers expect the slot to be unchanged at
; prologue_end, so the projection must not have a line 0 location, which would
; place prologue_end after it.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

define swifttailcc void @coroutine(ptr swiftasync %arg) "async_entry" !dbg !10 {
  %id = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr nonnull @coroutineTu)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)

  ; A suspend point with a line 0 location.
  %callee.ctx0 = call swiftcc ptr @swift_task_alloc(i64 16), !dbg !11
  %resume0 = call ptr @llvm.coro.async.resume(), !dbg !11
  %resume0.addr = getelementptr inbounds <{ ptr, ptr }>, ptr %callee.ctx0, i64 0, i32 1, !dbg !11
  store ptr %resume0, ptr %resume0.addr, align 8, !dbg !11
  %suspend0 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %resume0, ptr nonnull @__swift_async_resume_project_context, ptr nonnull @coroutine.0, ptr nonnull @callee, ptr nonnull %callee.ctx0), !dbg !12
  %resumed.ctx0 = extractvalue { ptr } %suspend0, 0, !dbg !12
  %caller.ctx0 = call ptr @__swift_async_resume_project_context(ptr %resumed.ctx0), !dbg !12
  call swiftcc void @swift_task_dealloc(ptr nonnull %callee.ctx0), !dbg !12
  call void @use(), !dbg !13
; CHECK-LABEL: define {{.*}} @coroutineTQ0_(
; CHECK:         load ptr, ptr %{{.*}}, !dbg ![[RESUME0_LOC:[0-9]+]]
; CHECK-NEXT:    call ptr @llvm.swift.async.context.addr(), !dbg ![[RESUME0_LOC]]
; CHECK-NEXT:    store ptr %{{.*}}, ptr %{{.*}}, !dbg ![[RESUME0_LOC]]

  ; A suspend point with a non-zero line location.
  %callee.ctx1 = call swiftcc ptr @swift_task_alloc(i64 16), !dbg !13
  %resume1 = call ptr @llvm.coro.async.resume(), !dbg !13
  %resume1.addr = getelementptr inbounds <{ ptr, ptr }>, ptr %callee.ctx1, i64 0, i32 1, !dbg !13
  store ptr %resume1, ptr %resume1.addr, align 8, !dbg !13
  %suspend1 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %resume1, ptr nonnull @__swift_async_resume_project_context, ptr nonnull @coroutine.0, ptr nonnull @callee, ptr nonnull %callee.ctx1), !dbg !14
  %resumed.ctx1 = extractvalue { ptr } %suspend1, 0, !dbg !14
  %caller.ctx1 = call ptr @__swift_async_resume_project_context(ptr %resumed.ctx1), !dbg !14
  call swiftcc void @swift_task_dealloc(ptr nonnull %callee.ctx1), !dbg !14
  call void @use(), !dbg !15
; CHECK-LABEL: define {{.*}} @coroutineTQ1_(
; CHECK:         load ptr, ptr %{{.*}}, !dbg ![[RESUME1_LOC:[0-9]+]]
; CHECK-NEXT:    call ptr @llvm.swift.async.context.addr(), !dbg ![[RESUME1_LOC]]
; CHECK-NEXT:    store ptr %{{.*}}, ptr %{{.*}}, !dbg ![[RESUME1_LOC]]

  call void (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 false, ptr nonnull @coroutine.0, ptr nonnull @callee, ptr %caller.ctx1), !dbg !15
  unreachable, !dbg !15
}

; CHECK-DAG: ![[RESUME0_LOC]] = !DILocation(line: 14,
; CHECK-DAG: ![[RESUME1_LOC]] = !DILocation(line: 20,

@coroutineTu = global <{i32, i32}> <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @coroutine to i64), i64 ptrtoint (ptr @coroutineTu to i64)) to i32), i32 16 }>, align 8

define linkonce_odr hidden ptr @__swift_async_resume_project_context(ptr %ctx) alwaysinline {
  %caller.ctx = load ptr, ptr %ctx, align 8
  %slot = call ptr @llvm.swift.async.context.addr()
  store ptr %caller.ctx, ptr %slot, align 8
  ret ptr %caller.ctx
}

define hidden swifttailcc void @coroutine.0(ptr %fn, ptr %ctx) {
  musttail call swifttailcc void %fn(ptr swiftasync %ctx)
  ret void
}

declare swifttailcc void @callee(ptr swiftasync)
declare void @use()
declare swiftcc ptr @swift_task_alloc(i64)
declare swiftcc void @swift_task_dealloc(ptr)
declare ptr @llvm.swift.async.context.addr()
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare ptr @llvm.coro.async.resume()
declare { ptr } @llvm.coro.suspend.async.sl_p0s(i32, ptr, ptr, ...)
declare void @llvm.coro.end.async(ptr, i1, ...)

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !2, producer: "", emissionKind: FullDebug)
!2 = !DIFile(filename: "main.swift", directory: "")
!3 = !DISubroutineType(types: null)
!10 = distinct !DISubprogram(name: "coroutine", linkageName: "coroutine", scope: !2, file: !2, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)
!11 = !DILocation(line: 5, scope: !10)
!12 = !DILocation(line: 0, scope: !10)
!13 = !DILocation(line: 14, column: 3, scope: !10)
!14 = !DILocation(line: 20, column: 3, scope: !10)
!15 = !DILocation(line: 21, column: 3, scope: !10)
