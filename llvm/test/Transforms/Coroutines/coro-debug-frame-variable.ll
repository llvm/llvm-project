; RUN: opt < %s -O0 -enable-coroutines -S | FileCheck %s
; RUN: opt < %s -passes='default<O0>' -enable-coroutines -S | FileCheck %s

; Define a function 'f' that resembles the Clang frontend's output for the
; following C++ coroutine:
;
;   void foo() {
;     int i = 0;
;     ++i;
;     print(i);  // Prints '1'
;
;     co_await suspend_always();
;
;     int j = 0;
;     ++i;
;     print(i);  // Prints '2'
;     ++j;
;     print(j);  // Prints '1'
;   }
;
; The CHECKs verify that dbg.declare intrinsics are created for the coroutine
; funclet 'f.resume', and that they reference the address of the variables on
; the coroutine frame. The debug locations for the original function 'f' are
; static (!11 and !13), whereas the coroutine funclet will have its own new
; ones with identical line and column numbers.
;
; CHECK-LABEL: define void @f() {
; CHECK:       entry:
; CHECK:         [[IGEP:%.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK:         [[JGEP:%.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 5
; CHECK:       init.ready:
; CHECK:         call void @llvm.dbg.declare(metadata i32* [[IGEP]], metadata ![[IVAR:[0-9]+]], metadata !DIExpression()), !dbg ![[IDBGLOC:[0-9]+]]
; CHECK:       await.ready:
; CHECK:         call void @llvm.dbg.declare(metadata i32* [[JGEP]], metadata ![[JVAR:[0-9]+]], metadata !DIExpression()), !dbg ![[JDBGLOC:[0-9]+]]
;
; CHECK-LABEL: define internal fastcc void @f.resume({{.*}}) {
; CHECK:       entry.resume:
; CHECK:         [[IGEP_RESUME:%.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK:         [[JGEP_RESUME:%.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 5
; CHECK:       init.ready:
; CHECK:         call void @llvm.dbg.declare(metadata i32* [[IGEP_RESUME]], metadata ![[IVAR_RESUME:[0-9]+]], metadata !DIExpression()), !dbg ![[IDBGLOC_RESUME:[0-9]+]]
; CHECK:       await.ready:
; CHECK:         call void @llvm.dbg.declare(metadata i32* [[JGEP_RESUME]], metadata ![[JVAR_RESUME:[0-9]+]], metadata !DIExpression()), !dbg ![[JDBGLOC_RESUME:[0-9]+]]
;
; CHECK: ![[IVAR]] = !DILocalVariable(name: "i"
; CHECK: ![[SCOPE:[0-9]+]] = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
; CHECK: ![[IDBGLOC]] = !DILocation(line: 24, column: 7, scope: ![[SCOPE]])
; CHECK: ![[JVAR]] = !DILocalVariable(name: "j"
; CHECK: ![[JDBGLOC]] = !DILocation(line: 32, column: 7, scope: ![[SCOPE]])
; CHECK: ![[IVAR_RESUME]] = !DILocalVariable(name: "i"
; CHECK: ![[RESUME_SCOPE:[0-9]+]] = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
; CHECK: ![[IDBGLOC_RESUME]] = !DILocation(line: 24, column: 7, scope: ![[RESUME_SCOPE]])
; CHECK: ![[JVAR_RESUME]] = !DILocalVariable(name: "j"
; CHECK: ![[JDBGLOC_RESUME]] = !DILocation(line: 32, column: 7, scope: ![[RESUME_SCOPE]])
define void @f() {
entry:
  %__promise = alloca i8, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
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
  call void @llvm.dbg.declare(metadata i32* %i, metadata !6, metadata !DIExpression()), !dbg !11
  store i32 0, i32* %i, align 4
  %i.init.ready.load = load i32, i32* %i, align 4
  %i.init.ready.inc = add nsw i32 %i.init.ready.load, 1
  store i32 %i.init.ready.inc, i32* %i, align 4
  %i.init.ready.reload = load i32, i32* %i, align 4
  call void @print(i32 %i.init.ready.reload)
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
  call void @llvm.dbg.declare(metadata i32* %j, metadata !12, metadata !DIExpression()), !dbg !13
  store i32 0, i32* %j, align 4
  %i.await.ready.load = load i32, i32* %i, align 4
  %i.await.ready.inc = add nsw i32 %i.await.ready.load, 1
  store i32 %i.await.ready.inc, i32* %i, align 4
  %j.await.ready.load = load i32, i32* %j, align 4
  %j.await.ready.inc = add nsw i32 %j.await.ready.load, 1
  store i32 %j.await.ready.inc, i32* %j, align 4
  %i.await.ready.reload = load i32, i32* %i, align 4
  call void @print(i32 %i.await.ready.reload)
  %j.await.ready.reload = load i32, i32* %j, align 4
  call void @print(i32 %j.await.ready.reload)
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

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i1 @llvm.coro.alloc(token)
declare i64 @llvm.coro.size.i64()
declare token @llvm.coro.save(i8*)
declare i8* @llvm.coro.begin(token, i8* writeonly)
declare i8 @llvm.coro.suspend(token, i1)
declare i8* @llvm.coro.free(token, i8* nocapture readonly)
declare i1 @llvm.coro.end(i8*, i1)

declare i8* @new(i64)
declare void @delete(i8*)
declare i1 @await_ready()
declare void @await_suspend()
declare void @await_resume()
declare void @print(i32)
declare i8* @from_address(i8*)
declare void @return_void()
declare void @final_suspend()

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
