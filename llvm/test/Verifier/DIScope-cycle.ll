; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; Reject a cycle in a DIScope parent chain (namespaces, modules, ...).

; CHECK: DIScope scope chain must not contain a cycle
; CHECK: !DINamespace(name: "a", scope: ![[NSB:[0-9]+]])
; CHECK: DIScope scope chain must not contain a cycle
; CHECK: !DIModule(scope: !{{[0-9]+}}, name: "m1"
; CHECK: warning: ignoring invalid debug info

@g = global i32 0, !dbg !13
@h = global i32 0, !dbg !16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: FullDebug, globals: !12)
!1 = !DIFile(filename: "scope-cycle.cpp", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DINamespace(name: "a", scope: !9)
!9 = !DINamespace(name: "b", scope: !8)
!10 = !DIModule(scope: !11, name: "m1")
!11 = !DIModule(scope: !10, name: "m2")

!12 = !{!13, !16}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "g", scope: !8, file: !1, line: 1, type: !7, isDefinition: true)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "h", scope: !10, file: !1, line: 2, type: !7, isDefinition: true)
