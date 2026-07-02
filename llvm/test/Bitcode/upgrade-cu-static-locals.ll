; Test moving of local static variables from DICompileUnit 'globals' to DISubprogram's 'retainedNodes'
;
; RUN: llvm-dis -o - %s.bc | FileCheck %s

; Function Attrs: norecurse nounwind
define dso_local i32 @foo() local_unnamed_addr #0 !dbg !2 {
  ret i32 0
}

define dso_local i32 @abc() local_unnamed_addr #0 !dbg !31 {
  ret i32 0
}

define dso_local i32 @xyz() local_unnamed_addr #0 !dbg !32 {
  ret i32 0
}

attributes #0 = { noinline optnone }

!llvm.dbg.cu = !{!7, !30}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

; CHECK: ![[CU:[0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C99, file: ![[F_TEST:[0-9]+]], producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !{{.*}}, globals: ![[CU_GLOBALS:[0-9]+]]
; CHECK: ![[CU_GLOBALS]] = !{![[AEXPR:[0-9]+]]}
; CHECK: ![[AEXPR]] = !DIGlobalVariableExpression(var: ![[A:[0-9]+]]
; CHECK: ![[A]] = distinct !DIGlobalVariable(name: "a", scope: ![[CU]]

; CHECK: ![[CU30:[0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C99, file: ![[F_TEST2:[0-9]+]], producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !{{.*}}, globals: ![[CU30_GLOBALS:[0-9]+]]
; CHECK: ![[CU30_GLOBALS]] = !{![[CAP_A_EXPR:[0-9]+]], ![[CAP_B_EXPR:[0-9]+]]}
; CHECK: ![[CAP_A_EXPR]] = !DIGlobalVariableExpression(var: ![[CAP_A:[0-9]+]]
; CHECK: ![[CAP_A]] = distinct !DIGlobalVariable(name: "A", scope: ![[CU30_GLOBALS:[0-9]+]]
; CHECK: ![[CAP_B_EXPR]] = !DIGlobalVariableExpression(var: ![[CAP_B:[0-9]+]]
; CHECK: ![[CAP_B]] = distinct !DIGlobalVariable(name: "B", scope: ![[CU30_GLOBALS:[0-9]+]]

; CHECK: ![[FOO:[0-9]+]] = distinct !DISubprogram(name: "foo", scope: ![[F_TEST]], file: ![[F_TEST]], line: 2, type: !{{.*}}, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[CU]], retainedNodes: ![[FOO_RETAINED:[0-9]+]])
; CHECK: ![[FOO_RETAINED]] = !{![[B_EXPR:[0-9]+]]}
; CHECK: ![[B_EXPR]] = !DIGlobalVariableExpression(var: ![[B:[0-9]+]]
; CHECK: ![[B]] = distinct !DIGlobalVariable(name: "b", scope: ![[FOO:[0-9]+]]

; CHECK: ![[ABC:[0-9]+]] = distinct !DISubprogram(name: "abc", scope: ![[CU30]], file: ![[F_TEST2]], line: 2, type: !{{.*}}, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[CU30]], retainedNodes: ![[ABC_RETAINED:[0-9]+]])
; CHECK: ![[ABC_RETAINED]] = !{![[ABC_A_EXPR:[0-9]+]], ![[ABC_B_EXPR:[0-9]+]]}
; CHECK: ![[ABC_A_EXPR]] = !DIGlobalVariableExpression(var: ![[ABC_A:[0-9]+]]
; CHECK: ![[ABC_A]] = distinct !DIGlobalVariable(name: "a", scope: ![[ABC]]
; CHECK: ![[ABC_B_EXPR]] = !DIGlobalVariableExpression(var: ![[ABC_B:[0-9]+]]
; CHECK: ![[ABC_B]] = distinct !DIGlobalVariable(name: "b", scope: ![[ABC]]

; CHECK: ![[XYZ:[0-9]+]] = distinct !DISubprogram(name: "xyz", scope: ![[CU30]], file: ![[F_TEST2]], line: 2, type: !{{.*}}, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[CU30]], retainedNodes: ![[XYZ_RETAINED:[0-9]+]])
; CHECK: ![[XYZ_RETAINED]] = !{![[XYZ_I:[0-9]+]], ![[XYZ_C_EXPR:[0-9]+]]}
; CHECK: ![[XYZ_I]] = !DILocalVariable(name: "i", arg: 1, scope: ![[XYZ]], file: ![[F_TEST2]]
; CHECK: ![[XYZ_C_EXPR]] = !DIGlobalVariableExpression(var: ![[XYZ_C:[0-9]+]]
; CHECK: ![[XYZ_C]] = distinct !DIGlobalVariable(name: "c", scope: ![[XYZ]]

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 2, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !7, retainedNodes: !8)
!3 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bug")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !9, nameTableKind: None)
!8 = !{}
!9 = !{!0, !10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "a", scope: !7, file: !3, line: 1, type: !12, isLocal: true, isDefinition: true)
!12 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !13)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !15)
!15 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 8.0.20181009 "}
!20 = !DILocation(line: 4, column: 10, scope: !2)
!21 = !{!22, !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 4, column: 14, scope: !2)
!25 = !{!26, !26, i64 0}
!26 = !{!"short", !22, i64 0}
!27 = !DILocation(line: 4, column: 12, scope: !2)

; Check CU with several global and several local static variables.
!30 = distinct !DICompileUnit(language: DW_LANG_C99, file: !50, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !45, nameTableKind: None)

!31 = distinct !DISubprogram(name: "abc", scope: !30, file: !50, line: 2, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !30, retainedNodes: !8)
!32 = distinct !DISubprogram(name: "xyz", scope: !30, file: !50, line: 2, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !30, retainedNodes: !46)

; Globals
!33 = !DIGlobalVariableExpression(var: !34, expr: !DIExpression())
!34 = distinct !DIGlobalVariable(name: "A", scope: !30, file: !50, line: 1, type: !12, isLocal: true, isDefinition: true)
!35 = !DIGlobalVariableExpression(var: !36, expr: !DIExpression())
!36 = distinct !DIGlobalVariable(name: "B", scope: !30, file: !50, line: 1, type: !12, isLocal: true, isDefinition: true)

; Static locals of abc
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = distinct !DIGlobalVariable(name: "a", scope: !31, file: !50, line: 3, type: !14, isLocal: true, isDefinition: true)
!39 = !DIGlobalVariableExpression(var: !40, expr: !DIExpression())
!40 = distinct !DIGlobalVariable(name: "b", scope: !31, file: !50, line: 3, type: !14, isLocal: true, isDefinition: true)

; Locals of xyz
!41 = !DIGlobalVariableExpression(var: !42, expr: !DIExpression())
!42 = distinct !DIGlobalVariable(name: "c", scope: !32, file: !50, line: 3, type: !14, isLocal: true, isDefinition: true)
!43 = !DILocalVariable(name: "i", line: 2, arg: 1, scope: !32, file: !50, type: !14)

!45 = !{!37, !33, !39, !35, !41}
!46 = !{!43}
!50 = !DIFile(filename: "test2.c", directory: "/home/yhs/work/tests/llvm/bug")
