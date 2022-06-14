; Tests whether we properly setup llvm.dbg.addr.
;
; Since we do not have any guarantees around the usage of llvm.dbg.addr, we can
; not propagate them like we do llvm.dbg.declare into funclets. But if users
; create the debug_value for us, make sure that we propagate llvm.dbg.addr into
; the beginning coroutine and all other funclets.

; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split,coro-split)' -S | FileCheck %s

; This file is based on coro-debug-frame-variable.ll.
; CHECK:  define internal fastcc void @f.resume(%f.Frame* noalias nonnull align 16 dereferenceable(80) %FramePtr) !dbg ![[RESUME_FN_DBG_NUM:[0-9]+]]
; CHECK-NEXT:       entry.resume:
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[XVAR_RESUME:[0-9]+]],
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[YVAR_RESUME:[0-9]+]],
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[ZVAR_RESUME:[0-9]+]],

; CHECK:  define internal fastcc void @f.destroy(%f.Frame* noalias nonnull align 16 dereferenceable(80) %FramePtr) !dbg ![[DESTROY_FN_DBG_NUM:[0-9]+]] {
; CHECK-NEXT:       entry.destroy:
; CHECK-NEXT:         %FramePtr.debug = alloca
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[XVAR_DESTROY:[0-9]+]],
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[YVAR_DESTROY:[0-9]+]],
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[ZVAR_DESTROY:[0-9]+]],

; CHECK: define internal fastcc void @f.cleanup(%f.Frame* noalias nonnull align 16 dereferenceable(80) %FramePtr) !dbg ![[CLEANUP_FN_DBG_NUM:[0-9]+]] {
; CHECK:       entry.cleanup:
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[XVAR_CLEANUP:[0-9]+]],
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[YVAR_CLEANUP:[0-9]+]],
; CHECK:         call void @llvm.dbg.addr(metadata %f.Frame** %FramePtr.debug, metadata ![[ZVAR_CLEANUP:[0-9]+]],

; CHECK-DAG: ![[RESUME_FN_DBG_NUM]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov"
; CHECK-DAG: ![[DESTROY_FN_DBG_NUM]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov"
; CHECK-DAG: ![[CLEANUP_FN_DBG_NUM]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov"
; CHECK-DAG: ![[XVAR_RESUME]] = !DILocalVariable(name: "x"
; CHECK-DAG: ![[YVAR_RESUME]] = !DILocalVariable(name: "y"
; CHECK-DAG: ![[ZVAR_RESUME]] = !DILocalVariable(name: "z"
; CHECK-DAG: ![[XVAR_DESTROY]] = !DILocalVariable(name: "x"
; CHECK-DAG: ![[YVAR_DESTROY]] = !DILocalVariable(name: "y"
; CHECK-DAG: ![[ZVAR_DESTROY]] = !DILocalVariable(name: "z"
; CHECK-DAG: ![[XVAR_CLEANUP]] = !DILocalVariable(name: "x"
; CHECK-DAG: ![[YVAR_CLEANUP]] = !DILocalVariable(name: "y"
; CHECK-DAG: ![[ZVAR_CLEANUP]] = !DILocalVariable(name: "z"

source_filename = "../llvm/test/Transforms/Coroutines/coro-debug-dbg.values-O2.ll"
declare void @consume(i32)

define void @f(i32 %i, i32 %j, i8* %ptr) presplitcoroutine !dbg !8 {
entry:
  %__promise = alloca i8, align 8
  %x = alloca [10 x i32], align 16
  %produced = call i32 @value_producer()
  %id = call token @llvm.coro.id(i32 16, i8* %__promise, i8* null, i8* null)
  %alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %alloc, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
  %size = call i64 @llvm.coro.size.i64()
  %memory = call i8* @new(i64 %size)
  br label %coro.init

coro.init:                                        ; preds = %coro.alloc, %entry
  %phi.entry.alloc = phi i8* [ null, %entry ], [ %memory, %coro.alloc ]
  %begin = call i8* @llvm.coro.begin(token %id, i8* %phi.entry.alloc)
  %ready = call i1 @await_ready()
  br i1 %ready, label %init.ready, label %init.suspend

init.suspend:                                     ; preds = %coro.init
  %save = call token @llvm.coro.save(i8* null)
  call void @await_suspend()
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %coro.ret [
    i8 0, label %init.ready
    i8 1, label %init.cleanup
  ]

init.cleanup:                                     ; preds = %init.suspend
  br label %cleanup

init.ready:                                       ; preds = %init.suspend, %coro.init
  call void @await_resume()
  %i.init.ready.inc = add nsw i32 0, 1
  call void @llvm.dbg.addr(metadata [10 x i32]* %x, metadata !12, metadata !DIExpression()), !dbg !17
  %memset = bitcast [10 x i32]* %x to i8*, !dbg !17
  call void @llvm.memset.p0i8.i64(i8* align 16 %memset, i8 0, i64 40, i1 false), !dbg !17
  call void @print(i32 %i.init.ready.inc)
  call void @llvm.dbg.addr(metadata i8* %ptr, metadata !24, metadata !DIExpression()), !dbg !17
  %ready.again = call zeroext i1 @await_ready()
  br i1 %ready.again, label %await.ready, label %await.suspend

await.suspend:                                    ; preds = %init.ready
  %save.again = call token @llvm.coro.save(i8* null)
  %from.address = call i8* @from_address(i8* %begin)
  call void @await_suspend()
  %suspend.again = call i8 @llvm.coro.suspend(token %save.again, i1 false)
  switch i8 %suspend.again, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %await.cleanup
  ]

await.cleanup:                                    ; preds = %await.suspend
  br label %cleanup

await.ready:                                      ; preds = %await.suspend, %init.ready
  call void @await_resume()
  %arrayidx0 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i64 0, i64 0, !dbg !19
  store i32 1, i32* %arrayidx0, align 16, !dbg !20
  call void @llvm.dbg.addr(metadata i32* %arrayidx0, metadata !18, metadata !DIExpression()), !dbg !11
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i64 0, i64 1, !dbg !21
  store i32 2, i32* %arrayidx1, align 4, !dbg !22
  %i.await.ready.inc = add nsw i32 %i.init.ready.inc, 1
  call void @consume(i32 %produced)
  call void @consume(i32 %i)
  call void @consume(i32 %j)
  call void @llvm.dbg.addr(metadata [10 x i32]* %x, metadata !23, metadata !DIExpression()), !dbg !17
  call void @print(i32 %i.await.ready.inc)
  call void @return_void()
  br label %coro.final

coro.final:                                       ; preds = %await.ready
  call void @final_suspend()
  %coro.final.await_ready = call i1 @await_ready()
  br i1 %coro.final.await_ready, label %final.ready, label %final.suspend

final.suspend:                                    ; preds = %coro.final
  %final.suspend.coro.save = call token @llvm.coro.save(i8* null)
  %final.suspend.from_address = call i8* @from_address(i8* %begin)
  call void @await_suspend()
  %final.suspend.coro.suspend = call i8 @llvm.coro.suspend(token %final.suspend.coro.save, i1 true)
  switch i8 %final.suspend.coro.suspend, label %coro.ret [
    i8 0, label %final.ready
    i8 1, label %final.cleanup
  ]

final.cleanup:                                    ; preds = %final.suspend
  br label %cleanup

final.ready:                                      ; preds = %final.suspend, %coro.final
  call void @await_resume()
  br label %cleanup

cleanup:                                          ; preds = %final.ready, %final.cleanup, %await.cleanup, %init.cleanup
  %cleanup.dest.slot.0 = phi i32 [ 0, %final.ready ], [ 2, %final.cleanup ], [ 2, %await.cleanup ], [ 2, %init.cleanup ]
  %free.memory = call i8* @llvm.coro.free(token %id, i8* %begin)
  %free = icmp ne i8* %free.memory, null
  br i1 %free, label %coro.free, label %after.coro.free

coro.free:                                        ; preds = %cleanup
  call void @delete(i8* %free.memory)
  br label %after.coro.free

after.coro.free:                                  ; preds = %coro.free, %cleanup
  switch i32 %cleanup.dest.slot.0, label %unreachable [
    i32 0, label %cleanup.cont
    i32 2, label %coro.ret
  ]

cleanup.cont:                                     ; preds = %after.coro.free
  br label %coro.ret

coro.ret:                                         ; preds = %cleanup.cont, %after.coro.free, %final.suspend, %await.suspend, %init.suspend
  %end = call i1 @llvm.coro.end(i8* null, i1 false)
  ret void

unreachable:                                      ; preds = %after.coro.free
  unreachable
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nounwind readnone
declare i64 @llvm.coro.size.i64() #3

; Function Attrs: nounwind
declare token @llvm.coro.save(i8*) #2

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #2

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: argmemonly nounwind readonly
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end(i8*, i1) #2

declare i8* @new(i64)

declare void @delete(i8*)

declare i1 @await_ready()

declare void @await_suspend()

declare void @await_resume()

declare void @print(i32)

declare i8* @from_address(i8*)

declare void @return_void()

declare void @final_suspend()

declare i32 @value_producer()

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #4

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

declare void @llvm.dbg.addr(metadata, metadata, metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { argmemonly nofree nosync nounwind willreturn writeonly }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "repro.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 11.0.0"}
!6 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 24, type: !10)
!7 = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 23, type: !9, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 0, scope: !7)
!12 = !DILocalVariable(name: "x", scope: !13, file: !1, line: 34, type: !14)
!13 = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 320, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 10)
!17 = !DILocation(line: 24, column: 7, scope: !7)
!18 = !DILocalVariable(name: "y", scope: !7, file: !1, line: 32, type: !10)
!19 = !DILocation(line: 42, column: 3, scope: !7)
!20 = !DILocation(line: 42, column: 8, scope: !7)
!21 = !DILocation(line: 43, column: 3, scope: !7)
!22 = !DILocation(line: 43, column: 8, scope: !7)
!23 = !DILocalVariable(name: "z", scope: !7, file: !1, line:24, type: !10)
!24 = !DILocalVariable(name: "ptr", scope: !7, file: !1, line: 34, type: !10)
