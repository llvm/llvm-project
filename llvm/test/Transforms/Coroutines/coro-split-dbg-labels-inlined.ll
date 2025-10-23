; RUN: opt %s -passes='cgscc(coro-split)' -S | FileCheck %s

source_filename = "coro.c"

; Function Attrs: nounwind uwtable
define ptr @f() #2 !dbg !16 {
entry:
  %0 = tail call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null), !dbg !26
  %1 = tail call i64 @llvm.coro.size.i64(), !dbg !26
  %frame = tail call ptr @malloc(i64 %1), !dbg !26
  %2 = tail call ptr @llvm.coro.begin(token %0, ptr %frame) #3, !dbg !26
  br label %loop1, !dbg !27

loop1:
  %3 = tail call token @llvm.coro.save(ptr null), !dbg !34
  %4 = tail call i8 @llvm.coro.suspend(token %3, i1 false), !dbg !34
  switch i8 %4, label %coro_Suspend [
    i8 0, label %loop1
    i8 1, label %coro_Cleanup
  ], !dbg !34

coro_Cleanup:
  %7 = tail call ptr @llvm.coro.free(token %0, ptr %2), !dbg !37
  tail call void @free(ptr nonnull %7), !dbg !37
  br label %coro_Suspend, !dbg !37

coro_Suspend:
  tail call i1 @llvm.coro.end(ptr null, i1 false, token none) #3, !dbg !40
  ret ptr %2, !dbg !41
}

; Check that the resume function contains the `#dbg_label` instructions.
; CHECK-LABEL:   define internal fastcc void @f.resume({{.*}})
; CHECK-SAME:      !dbg ![[RESUME_SUBPROGRAM:[0-9]+]]
; CHECK:         resume.0:
; CHECK-NEXT:        #dbg_label(![[RESUME_0:[0-9]+]], ![[RESUME_LABEL_LOC:[0-9]+]])

; Check that the destroy function contains the `#dbg_label` instructions.
; CHECK-LABEL:   define internal fastcc void @f.destroy({{.*}})
; CHECK-SAME:      !dbg ![[DESTROY_SUBPROGRAM:[0-9]+]]
; CHECK:         resume.0:
; CHECK-NEXT:        #dbg_label(![[DESTROY_0:[0-9]+]], ![[DESTROY_LABEL_LOC:[0-9]+]])

; Check that the DILabels are correctly based to their "inlined at" location.
; CHECK: ![[RESUME_LABEL_LOC]] = !DILocation(line: 12, column: 6, scope: ![[RESUME_SUBPROGRAM]])
; CHECK: ![[RESUME_0]] = !DILabel(scope: ![[RESUME_SUBPROGRAM]], name: "__coro_resume_0", file: !{{[0-9]*}}, line: 42, column: 2, isArtificial: true, coroSuspendIdx: 0)

; CHECK: ![[DESTROY_LABEL_LOC]] = !DILocation(line: 12, column: 6, scope: ![[DESTROY_SUBPROGRAM]])
; CHECK: ![[DESTROY_0]] = !DILabel(scope: ![[DESTROY_SUBPROGRAM]], name: "__coro_resume_0", file: !{{[0-9]*}}, line: 42, column: 2, isArtificial: true, coroSuspendIdx: 0)

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #4

declare noalias ptr @malloc(i64) local_unnamed_addr #0
declare i64 @llvm.coro.size.i64() #1
declare ptr @llvm.coro.begin(token, ptr writeonly) #0
declare token @llvm.coro.save(ptr) #0
declare i8 @llvm.coro.suspend(token, i1) #0
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #4
declare void @free(ptr nocapture) local_unnamed_addr #0
declare i1 @llvm.coro.end(ptr, i1, token) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind uwtable presplitcoroutine }
attributes #3 = { noduplicate }
attributes #4 = { argmemonly nounwind readonly }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 - manually edited", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "coro.c", directory: "/home/gor/build/bin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 - manually edited"}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!16 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 8, type: !17, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !0, retainedNodes: !20)
!17 = !DISubroutineType(types: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!20 = !{!21, !22, !24}
!21 = !DILocalVariable(name: "coro_hdl", scope: !16, file: !1, line: 9, type: !19)
!22 = !DILocalVariable(name: "i", scope: !23, file: !1, line: 11, type: !9)
!23 = distinct !DILexicalBlock(scope: !16, file: !1, line: 11, column: 3)
!24 = !DILocalVariable(name: "coro_mem", scope: !16, file: !1, line: 16, type: !19)
!26 = !DILocation(line: 9, column: 3, scope: !16)
!27 = !DILocation(line: 10, column: 8, scope: !23)
!33 = !DILocation(line: 11, column: 6, scope: !23)
!34 = !DILocation(line: 42, column: 2, scope: !100, inlinedAt: !101)
!35 = !DILocation(line: 13, column: 6, scope: !23)
!36 = !DILocation(line: 14, column: 6, scope: !23)
!37 = !DILocation(line: 16, column: 3, scope: !16)
!40 = !DILocation(line: 16, column: 3, scope: !16)
!41 = !DILocation(line: 17, column: 1, scope: !16)

!100 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 8, type: !17, scopeLine: 4, isOptimized: true, unit: !0, retainedNodes: !2)
!101 = !DILocation(line: 12, column: 6, scope: !16)
