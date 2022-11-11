; Test that llvm-reduce can drop unneeded debug metadata nodes referenced by
; DICompileUnit and DISuprogram.
;
; RUN: llvm-reduce --delta-passes=di-metadata --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck <%t --enable-var-scope %s

; CHECK-INTERESTINGNESS: define void @test() !dbg [[SUBPROG:![0-9]+]]
; CHECK-INTERESTINGNESS: !llvm.module.flags = !{

; CHECK-INTERESTINGNESS: !llvm.dbg.cu = !{[[CU:.+]]}

; CHECK-INTERESTINGNESS-DAG: [[CU]] = distinct !DICompileUnit(language: DW_LANG_C99,{{.*}}, retainedTypes: [[TYPES:![0-9]+]]
; CHECK-INTERESTINGNESS-DAG: [[TYPES]] = !{[[T0:![0-9]+]]
; CHECK-INTERESTINGNESS-DAG: [[T0]] = !DIBasicType(name: "unsigned int",
; CHECK-INTERESTINGNESS-DAG: [[SUBPROG]] = distinct !DISubprogram(name: "test",



; CHECK: define void @test() !dbg [[SUBPROG:![0-9]+]]
; CHECK: !llvm.module.flags = !{

; CHECK: !llvm.dbg.cu = !{[[CU:.+]]}

; CHECK-DAG: [[CU]] = distinct !DICompileUnit(language: DW_LANG_C99,{{.*}}, retainedTypes: [[TYPES:![0-9]+]], globals: [[GLOBALS:![0-9]+]]
; CHECK-DAG: [[EMPTY:![0-9]+]] = !{}
; CHECK-DAG: [[TYPES]] = !{[[T0:![0-9]+]]
; CHECK-DAG: [[T0]] = !DIBasicType(name: "unsigned int",
; CHECK-DAG: [[GLOBALS]] = !{{{![0-9]+}}

; CHECK-DAG: [[SUBPROG]] = distinct !DISubprogram(name: "test", {{.*}}retainedNodes: [[EMPTY]])

define void @test() !dbg !17 {
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !12, globals: !14, splitDebugInlining: false, nameTableKind: None, sysroot: "/", sdk: "SDK")
!3 = !DIFile(filename: "test.c", directory: "/tmp")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !6, line: 755, baseType: !7, size: 32, elements: !8)
!6 = !DIFile(filename: "foo.h", directory: "/tmp")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11}
!9 = !DIEnumerator(name: "flag_AUTO", value: 0)
!10 = !DIEnumerator(name: "flag_STDIN", value: 1)
!11 = !DIEnumerator(name: "flag_INTERACTIVE", value: 2)
!12 = !{!7, !13}
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned)
!14 = !{!15}
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(scope: null, file: !3, line: 726, type: !13, isLocal: true, isDefinition: true)
!17 = distinct !DISubprogram(name: "test", scope: !18, file: !18, line: 1839, type: !19, scopeLine: 1846, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!18 = !DIFile(filename: "bar.c", directory: "/tmp")
!19 = !DISubroutineType(types: !20)
!20 = !{null, !7}
!21 = !{!22, !23, !24}
!22 = !DILocalVariable(name: "A", arg: 1, scope: !17, file: !18, line: 1839, type: !7)
!23 = !DILocalVariable(name: "B", arg: 2, scope: !17, file: !18, line: 1839, type: !7)
!24 = !DILocalVariable(name: "C", scope: !17, file: !18, line: 1847, type: !7)
