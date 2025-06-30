; RUN: opt < %s -passes='default<O0>' -S | FileCheck %s

; Define a function 'f' that resembles the Clang frontend's output for the
; following C++ coroutine:
;
;   void foo() {
;     int i = 0;
;     ++i;
;     int x = {};
;     print(i);  // Prints '1'
;
;     co_await suspend_always();
;
;     int j = 0;
;     x[0] = 1;
;     x[1] = 2;
;     ++i;
;     print(i);  // Prints '2'
;     ++j;
;     print(j);  // Prints '1'
;     print(x);  // Print '1'
;   }
;
; The CHECKs verify that dbg.declare intrinsics are created for the coroutine
; funclet 'f.resume', and that they reference the address of the variables on
; the coroutine frame. The debug locations for the original function 'foo' are
; static (!11 and !13), whereas the coroutine funclet will have its own new
; ones with identical line and column numbers.
;
; CHECK-LABEL: define void @_Z3foov() {{.*}} {
; CHECK:       entry:
; CHECK:         %j = alloca i32, align 4
; CHECK:         #dbg_declare(ptr %j, ![[JVAR:[0-9]+]], !DIExpression(), ![[JDBGLOC:[0-9]+]]
; CHECK:         %[[MEMORY:.*]] = call ptr @new({{.+}}), !dbg ![[IDBGLOC:[0-9]+]]
; CHECK:         #dbg_declare(ptr %[[MEMORY]], ![[XVAR:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 32), ![[IDBGLOC]]
; CHECK:         #dbg_declare(ptr %[[MEMORY]], ![[IVAR:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 20), ![[IDBGLOC]]
; CHECK:       await.ready:
;
; CHECK-LABEL: define internal fastcc void @_Z3foov.resume({{.*}}) {{.*}} {
; CHECK:       entry.resume:
; CHECK-NEXT:    %[[DBG_PTR:.*]] = alloca ptr
; CHECK-NEXT:    #dbg_declare(ptr %[[DBG_PTR]], ![[XVAR_RESUME:[0-9]+]],   !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 32),
; CHECK-NEXT:    #dbg_declare(ptr %[[DBG_PTR]], ![[IVAR_RESUME:[0-9]+]], !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 20), ![[IDBGLOC_RESUME:[0-9]+]]
; CHECK-NEXT:    #dbg_declare(ptr %[[DBG_PTR]], ![[FRAME_RESUME:[0-9]+]], !DIExpression(DW_OP_deref),
; CHECK-NEXT:    store ptr {{.*}}, ptr %[[DBG_PTR]]
; CHECK:         %[[J:.*]] = alloca i32, align 4
; CHECK-NEXT:    #dbg_declare(ptr %[[J]], ![[JVAR_RESUME:[0-9]+]], !DIExpression(), ![[JDBGLOC_RESUME:[0-9]+]]
; CHECK:       init.ready:
; CHECK:       await.ready:
;
; CHECK-DAG: ![[FRAME_RESUME]] = !DILocalVariable(name: "__coro_frame"
; CHECK-DAG: ![[IVAR]] = !DILocalVariable(name: "i"
; CHECK-DAG: ![[PROG_SCOPE:[0-9]+]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov"
; CHECK-DAG: ![[BLK_SCOPE:[0-9]+]] = distinct !DILexicalBlock(scope: ![[PROG_SCOPE]], file: !1, line: 23, column: 12)
; CHECK-DAG: ![[IDBGLOC]] = !DILocation(line: 23, column: 6, scope: ![[PROG_SCOPE]])
; CHECK-DAG: ![[XVAR]] = !DILocalVariable(name: "x"
; CHECK-DAG: ![[JVAR]] = !DILocalVariable(name: "j"
; CHECK-DAG: ![[JDBGLOC]] = !DILocation(line: 32, column: 7, scope: ![[BLK_SCOPE]])

; CHECK-DAG: ![[XVAR_RESUME]] = !DILocalVariable(name: "x"
; CHECK-DAG: ![[RESUME_PROG_SCOPE:[0-9]+]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov.resume"
; CHECK-DAG: ![[IDBGLOC_RESUME]] = !DILocation(line: 24, column: 7, scope: ![[RESUME_BLK_SCOPE:[0-9]+]])
; CHECK-DAG: ![[RESUME_BLK_SCOPE]] = distinct !DILexicalBlock(scope: ![[RESUME_PROG_SCOPE]], file: !1, line: 23, column: 12)
; CHECK-DAG: ![[IVAR_RESUME]] = !DILocalVariable(name: "i"
; CHECK-DAG: ![[JVAR_RESUME]] = !DILocalVariable(name: "j"
; CHECK-DAG: ![[JDBGLOC_RESUME]] = !DILocation(line: 32, column: 7, scope: ![[RESUME_BLK_SCOPE]])
define void @_Z3foov() presplitcoroutine !dbg !8 {
entry:
  %__promise = alloca i8, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %x = alloca [10 x i32], align 16
  %id = call token @llvm.coro.id(i32 16, ptr %__promise, ptr null, ptr null)
  %alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %alloc, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
  %size = call i64 @llvm.coro.size.i64(), !dbg !23
  %memory = call ptr @new(i64 %size), !dbg !23
  br label %coro.init, !dbg !23

coro.init:                                        ; preds = %coro.alloc, %entry
  %phi.entry.alloc = phi ptr [ null, %entry ], [ %memory, %coro.alloc ], !dbg !23
  %begin = call ptr @llvm.coro.begin(token %id, ptr %phi.entry.alloc), !dbg !23
  %ready = call i1 @await_ready()
  br i1 %ready, label %init.ready, label %init.suspend

init.suspend:                                     ; preds = %coro.init
  %save = call token @llvm.coro.save(ptr null)
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
  call void @llvm.dbg.declare(metadata ptr %i, metadata !6, metadata !DIExpression()), !dbg !11
  store i32 0, ptr %i, align 4
  %i.init.ready.load = load i32, ptr %i, align 4
  %i.init.ready.inc = add nsw i32 %i.init.ready.load, 1
  store i32 %i.init.ready.inc, ptr %i, align 4
  call void @llvm.dbg.declare(metadata ptr %x, metadata !14, metadata !DIExpression()), !dbg !11
  call void @llvm.memset.p0.i64(ptr align 16 %x, i8 0, i64 40, i1 false), !dbg !11
  %i.init.ready.reload = load i32, ptr %i, align 4
  call void @print(i32 %i.init.ready.reload)
  %ready.again = call zeroext i1 @await_ready()
  br i1 %ready.again, label %await.ready, label %await.suspend

await.suspend:                                    ; preds = %init.ready
  %save.again = call token @llvm.coro.save(ptr null)
  %from.address = call ptr @from_address(ptr %begin)
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
  call void @llvm.dbg.declare(metadata ptr %j, metadata !12, metadata !DIExpression()), !dbg !13
  store i32 0, ptr %j, align 4
  store i32 1, ptr %x, align 16, !dbg !19
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr %x, i64 0, i64 1, !dbg !20
  store i32 2, ptr %arrayidx1, align 4, !dbg !21
  %i.await.ready.load = load i32, ptr %i, align 4
  %i.await.ready.inc = add nsw i32 %i.await.ready.load, 1
  store i32 %i.await.ready.inc, ptr %i, align 4
  %j.await.ready.load = load i32, ptr %j, align 4
  %j.await.ready.inc = add nsw i32 %j.await.ready.load, 1
  store i32 %j.await.ready.inc, ptr %j, align 4
  %i.await.ready.reload = load i32, ptr %i, align 4
  call void @print(i32 %i.await.ready.reload)
  %j.await.ready.reload = load i32, ptr %j, align 4
  call void @print(i32 %j.await.ready.reload)
  call void @return_void()
  br label %coro.final

coro.final:                                       ; preds = %await.ready
  call void @final_suspend()
  %coro.final.await_ready = call i1 @await_ready()
  br i1 %coro.final.await_ready, label %final.ready, label %final.suspend

final.suspend:                                    ; preds = %coro.final
  %final.suspend.coro.save = call token @llvm.coro.save(ptr null)
  %final.suspend.from_address = call ptr @from_address(ptr %begin)
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
  %free.memory = call ptr @llvm.coro.free(token %id, ptr %begin)
  %free = icmp ne ptr %free.memory, null
  br i1 %free, label %coro.free, label %after.coro.free

coro.free:                                        ; preds = %cleanup
  call void @delete(ptr %free.memory)
  br label %after.coro.free

after.coro.free:                                  ; preds = %coro.free, %cleanup
  switch i32 %cleanup.dest.slot.0, label %unreachable [
    i32 0, label %cleanup.cont
    i32 2, label %coro.ret
  ]

cleanup.cont:                                     ; preds = %after.coro.free
  br label %coro.ret

coro.ret:                                         ; preds = %cleanup.cont, %after.coro.free, %final.suspend, %await.suspend, %init.suspend
  %end = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void

unreachable:                                      ; preds = %after.coro.free
  unreachable
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i1 @llvm.coro.alloc(token)
declare i64 @llvm.coro.size.i64()
declare token @llvm.coro.save(ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr nocapture readonly)
declare i1 @llvm.coro.end(ptr, i1, token)

declare ptr @new(i64)
declare void @delete(ptr)
declare i1 @await_ready()
declare void @await_suspend()
declare void @await_resume()
declare void @print(i32)
declare ptr @from_address(ptr)
declare void @return_void()
declare void @final_suspend()

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

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
!11 = !DILocation(line: 24, column: 7, scope: !7)
!12 = !DILocalVariable(name: "j", scope: !7, file: !1, line: 32, type: !10)
!13 = !DILocation(line: 32, column: 7, scope: !7)
!14 = !DILocalVariable(name: "x", scope: !22, file: !1, line: 34, type: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 320, elements: !16)
!16 = !{!17}
!17 = !DISubrange(count: 10)
!18 = !DILocation(line: 42, column: 3, scope: !7)
!19 = !DILocation(line: 42, column: 8, scope: !7)
!20 = !DILocation(line: 43, column: 3, scope: !7)
!21 = !DILocation(line: 43, column: 8, scope: !7)
!22 = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
!23 = !DILocation(line: 23, column: 6, scope: !8)
