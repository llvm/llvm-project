; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split,coro-split)' -S | FileCheck %s

; Checks whether the dbg.declare for `__coro_frame` are created.

; CHECK-LABEL: define void @f(
; CHECK:       coro.init:
; CHECK:        %[[begin:.*]] = call noalias nonnull i8* @llvm.coro.begin(
; CHECK:        call void @llvm.dbg.declare(metadata i8* %[[begin]], metadata ![[CORO_FRAME:[0-9]+]], metadata !DIExpression())
; CHECK:        %[[FramePtr:.*]] = bitcast i8* %[[begin]] to
;
; CHECK:       define internal fastcc void @f.resume(
; CHECK:       entry.resume:
; CHECK:            call void @llvm.dbg.declare(metadata %f.Frame** %[[FramePtr_RESUME:.*]], metadata ![[CORO_FRAME_IN_RESUME:[0-9]+]], metadata !DIExpression()
;
; CHECK-DAG: ![[FILE:[0-9]+]] = !DIFile(filename: "coro-debug.cpp"
; CHECK-DAG: ![[RAMP:[0-9]+]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov",
; CHECK-DAG: ![[RAMP_SCOPE:[0-9]+]] = distinct !DILexicalBlock(scope: ![[RAMP]], file: ![[FILE]], line: 23
; CHECK-DAG: ![[CORO_FRAME]] = !DILocalVariable(name: "__coro_frame", scope: ![[RAMP_SCOPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE:[0-9]+]], type: ![[FRAME_TYPE:[0-9]+]], flags: DIFlagArtificial)
; CHECK-DAG: ![[FRAME_TYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "__coro_frame_ty", scope: ![[RAMP]], {{.*}}elements: ![[ELEMENTS:[0-9]+]]
; CHECK-DAG: ![[ELEMENTS]] = !{![[RESUME_FN:[0-9]+]], ![[DESTROY_FN:[0-9]+]], ![[PROMISE:[0-9]+]], ![[INT64_0:[0-9]+]], ![[DOUBLE_1:[0-9]+]], ![[INT32_2:[0-9]+]], ![[INT32_3:[0-9]+]], ![[STRUCT_4:[0-9]+]], ![[CORO_INDEX:[0-9]+]]
; CHECK-DAG: ![[RESUME_FN]] = !DIDerivedType(tag: DW_TAG_member, name: "__resume_fn"{{.*}}, flags: DIFlagArtificial
; CHECK-DAG: ![[DESTROY_FN]] = !DIDerivedType(tag: DW_TAG_member, name: "__destroy_fn"{{.*}}, flags: DIFlagArtificial
; CHECK-DAG: ![[PROMISE]] = !DIDerivedType(tag: DW_TAG_member, name: "__promise",{{.*}}baseType: ![[PROMISE_BASE:[0-9]+]]
; CHECK-DAG: ![[PROMISE_BASE]] = !DIDerivedType(tag: DW_TAG_typedef, name: "promise_type"
; CHECK-DAG: ![[INT64_0]] = !DIDerivedType(tag: DW_TAG_member, name: "__int_64_0", scope: ![[FRAME_TYPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE]], baseType: ![[I64_BASE:[0-9]+]],{{.*}}, flags: DIFlagArtificial
; CHECK-DAG: ![[I64_BASE]] = !DIBasicType(name: "__int_64", size: 64, encoding: DW_ATE_signed, flags: DIFlagArtificial)
; CHECK-DAG: ![[DOUBLE_1]] = !DIDerivedType(tag: DW_TAG_member, name: "__double__1", scope: ![[FRAME_TYPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE]], baseType: ![[DOUBLE_BASE:[0-9]+]]{{.*}}, flags: DIFlagArtificial
; CHECK-DAG: ![[DOUBLE_BASE]] = !DIBasicType(name: "__double_", size: 64, encoding: DW_ATE_float, flags: DIFlagArtificial)
; CHECK-DAG: ![[INT32_2]] = !DIDerivedType(tag: DW_TAG_member, name: "__int_32_2", scope: ![[FRAME_TYPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE]], baseType: ![[I32_BASE:[0-9]+]]{{.*}}, flags: DIFlagArtificial
; CHECK-DAG: ![[I32_BASE]] = !DIBasicType(name: "__int_32", size: 32, encoding: DW_ATE_signed, flags: DIFlagArtificial)
; CHECK-DAG: ![[INT32_3]] = !DIDerivedType(tag: DW_TAG_member, name: "__int_32_3", scope: ![[FRAME_TYPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE]], baseType: ![[I32_BASE]]
; CHECK-DAG: ![[STRUCT_4]] = !DIDerivedType(tag: DW_TAG_member, name: "struct_big_structure_4", scope: ![[FRAME_TYPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE]], baseType: ![[STRUCT_BASE:[0-9]+]]
; CHECK-DAG: ![[STRUCT_BASE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "struct_big_structure"{{.*}}elements: ![[STRUCT_ELEMENTS:[0-9]+]]
; CHECK-DAG: ![[STRUCT_ELEMENTS]] = !{![[MEM_TYPE:[0-9]+]]}
; CHECK-DAG: ![[MEM_TYPE]] = !DIDerivedType(tag: DW_TAG_member, name: "UnknownType_4000"
; CHECK-DAG: ![[CORO_INDEX]] = !DIDerivedType(tag: DW_TAG_member, name: "__coro_index"
; CHECK-DAG: ![[PROMISE_VAR:[0-9]+]] = !DILocalVariable(name: "__promise", scope: ![[RAMP_SCOPE]], file: ![[FILE]], line: [[PROMISE_VAR_LINE]]
; CHECK-DAG: ![[BAR_FUNC:[0-9]+]] = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv",
; CHECK-DAG: ![[BAR_SCOPE:[0-9]+]] = distinct !DILexicalBlock(scope: ![[BAR_FUNC]], file: !1
; CHECK-DAG: ![[FRAME_TYPE_IN_BAR:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "__coro_frame_ty", scope: ![[BAR_FUNC]], file: ![[FILE]], line: [[BAR_LINE:[0-9]+]]{{.*}}elements: ![[ELEMENTS_IN_BAR:[0-9]+]]
; CHECK-DAG: ![[ELEMENTS_IN_BAR]] = !{![[RESUME_FN_IN_BAR:[0-9]+]], ![[DESTROY_FN_IN_BAR:[0-9]+]], ![[PROMISE_IN_BAR:[0-9]+]], ![[INT64_0_IN_BAR:[0-9]+]], ![[DOUBLE_1_IN_BAR:[0-9]+]], ![[INT32_2_IN_BAR:[0-9]+]], ![[STRUCT_3_IN_BAR:[0-9]+]], ![[CORO_INDEX_IN_BAR:[0-9]+]]
; CHECK-DAG: ![[PROMISE_IN_BAR]] = !DIDerivedType(tag: DW_TAG_member, name: "__promise",{{.*}}baseType: ![[PROMISE_BASE]]
; CHECK-DAG: ![[INT64_0_IN_BAR]] = !DIDerivedType(tag: DW_TAG_member, name: "__int_64_0", scope: ![[FRAME_TYPE_IN_BAR]], file: ![[FILE]], line: [[BAR_LINE]], baseType: ![[I64_BASE]]
; CHECK-DAG: ![[DOUBLE_1_IN_BAR]] = !DIDerivedType(tag: DW_TAG_member, name: "__double__1", scope: ![[FRAME_TYPE_IN_BAR]], file: ![[FILE]], line: [[BAR_LINE]], baseType: ![[DOUBLE_BASE]]
; CHECK-DAG: ![[INT32_2_IN_BAR]] = !DIDerivedType(tag: DW_TAG_member, name: "__int_32_2", scope: ![[FRAME_TYPE_IN_BAR]], file: ![[FILE]], line: [[BAR_LINE]], baseType: ![[I32_BASE]]
; CHECK-DAG: ![[STRUCT_3_IN_BAR]] = !DIDerivedType(tag: DW_TAG_member, name: "struct_big_structure_3", scope: ![[FRAME_TYPE_IN_BAR]], file: ![[FILE]], line: [[BAR_LINE]], baseType: ![[STRUCT_BASE_IN_BAR:[0-9]+]]
; CHECK-DAG: ![[STRUCT_BASE_IN_BAR]] = !DICompositeType(tag: DW_TAG_structure_type, name: "struct_big_structure", scope: ![[FRAME_TYPE_IN_BAR]], file: ![[FILE]], line: [[BAR_LINE]],{{.*}}
; CHECK-DAG: ![[CORO_FRAME_IN_RESUME]] = !DILocalVariable(name: "__coro_frame",{{.*}}type: ![[FRAME_TYPE]]


%promise_type = type { i32, i32, double }
%struct.big_structure = type { [500 x i8] }
declare void @produce(%struct.big_structure*)
declare void @consume(%struct.big_structure*)
declare void @pi32(i32*)
declare void @pi64(i64*)
declare void @pdouble(double*)

define void @f(i32 %a, i32 %b, i64 %c, double %d) "coroutine.presplit"="0" !dbg !8 {
entry:
    %__promise = alloca %promise_type, align 8
    %0 = bitcast %promise_type* %__promise to i8*
    %a.alloc = alloca i32, align 4
    %b.alloc = alloca i32, align 4
    %c.alloc = alloca i64, align 4
    %d.alloc = alloca double, align 4
    store i32 %a, i32* %a.alloc
    store i32 %b, i32* %b.alloc
    store i64 %c, i64* %c.alloc
    store double %d, double* %d.alloc
    %struct.data = alloca %struct.big_structure, align 1
    call void @produce(%struct.big_structure* %struct.data)
    %id = call token @llvm.coro.id(i32 16, i8* %0, i8* null, i8* null)
    %alloc = call i1 @llvm.coro.alloc(token %id)
    br i1 %alloc, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
    %size = call i64 @llvm.coro.size.i64()
    %memory = call i8* @new(i64 %size)
    br label %coro.init

coro.init:                                        ; preds = %coro.alloc, %entry
    %phi.entry.alloc = phi i8* [ null, %entry ], [ %memory, %coro.alloc ]
    %begin = call i8* @llvm.coro.begin(token %id, i8* %phi.entry.alloc)
    call void @llvm.dbg.declare(metadata %promise_type* %__promise, metadata !6, metadata !DIExpression()), !dbg !18
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
    %i.i = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 0
    store i32 1, i32* %i.i, align 8
    %j.i = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 1
    store i32 2, i32* %j.i, align 4
    %k.i = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 2
    store double 3.000000e+00, double* %k.i, align 8
    call void @consume(%struct.big_structure* %struct.data)
    call void @pi32(i32* %a.alloc)
    call void @pi32(i32* %b.alloc)
    call void @pi64(i64* %c.alloc)
    call void @pdouble(double* %d.alloc)
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

define void @bar(i32 %a, i64 %c, double %d) "coroutine.presplit"="0" !dbg !19 {
entry:
    %__promise = alloca %promise_type, align 8
    %0 = bitcast %promise_type* %__promise to i8*
    %a.alloc = alloca i32, align 4
    %c.alloc = alloca i64, align 4
    %d.alloc = alloca double, align 4
    store i32 %a, i32* %a.alloc
    store i64 %c, i64* %c.alloc
    store double %d, double* %d.alloc
    %struct.data = alloca %struct.big_structure, align 1
    call void @produce(%struct.big_structure* %struct.data)
    %id = call token @llvm.coro.id(i32 16, i8* %0, i8* null, i8* null)
    %alloc = call i1 @llvm.coro.alloc(token %id)
    br i1 %alloc, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %entry
    %size = call i64 @llvm.coro.size.i64()
    %memory = call i8* @new(i64 %size)
    br label %coro.init

coro.init:                                        ; preds = %coro.alloc, %entry
    %phi.entry.alloc = phi i8* [ null, %entry ], [ %memory, %coro.alloc ]
    %begin = call i8* @llvm.coro.begin(token %id, i8* %phi.entry.alloc)
    call void @llvm.dbg.declare(metadata %promise_type* %__promise, metadata !21, metadata !DIExpression()), !dbg !22
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
     %i.i = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 0
    store i32 1, i32* %i.i, align 8
    %j.i = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 1
    store i32 2, i32* %j.i, align 4
    %k.i = getelementptr inbounds %promise_type, %promise_type* %__promise, i64 0, i32 2
    store double 3.000000e+00, double* %k.i, align 8
    call void @consume(%struct.big_structure* %struct.data)
    call void @pi32(i32* %a.alloc)
    call void @pi64(i64* %c.alloc)
    call void @pdouble(double* %d.alloc)
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
!1 = !DIFile(filename: "coro-debug.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 11.0.0"}
!6 = !DILocalVariable(name: "__promise", scope: !7, file: !1, line: 24, type: !10)
!7 = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !8, file: !1, line: 23, type: !9, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "promise_type", scope: !8, file: !1, line: 15, baseType: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "promise_type", scope: !8, file: !1, line: 10, size: 128, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !12, identifier: "_ZTSN4coro12promise_typeE")
!12 = !{!13, !14, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !8, file: !1, line: 10, baseType: !16, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !8, file: !1, line: 10, baseType: !16, size: 32, offset: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "k", scope: !8, file: !1, line: 10, baseType: !17, size: 64, offset: 64)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!18 = !DILocation(line: 8, scope: !7)
!19 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !19, file: !1, line: 54, type: !9, scopeLine: 54, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!20 = distinct !DILexicalBlock(scope: !19, file: !1, line: 23, column: 12)
!21 = !DILocalVariable(name: "__promise", scope: !20, file: !1, line: 55, type: !10)
!22 = !DILocation(line: 10, scope: !20)






