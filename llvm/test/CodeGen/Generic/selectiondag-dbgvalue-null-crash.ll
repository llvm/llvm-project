; RUN: llc -O3 < %s
;
; Regression test for a null pointer dereference in 
; SelectionDAG::resolveDanglingDebugInfo when Val.getNode() returns null
; for aggregate types with nested empty structs.
;
; The crash occurred when:
; 1. A dbg_value references an aggregate type containing empty structs {}
; 2. An insertvalue operation on such types gets lowered by SelectionDAG
; 3. The resulting SDValue has a null node, causing a crash when accessed

define void @test() !dbg !4 {
entry:
  %tmp = alloca { { i1, {} }, ptr, { { {} }, { {} } }, i64 }, align 8
    #dbg_value({ { {} }, { {} } } zeroinitializer, !5, !DIExpression(), !6)
    #dbg_value(i64 2, !7, !DIExpression(), !6)
  %0 = insertvalue { { i1, {} }, ptr, { { {} }, { {} } }, i64 } { { i1, {} } zeroinitializer, ptr null, { { {} }, { {} } } zeroinitializer, i64 2 }, ptr null, 1, !dbg !6
  %1 = insertvalue { { i1, {} }, ptr, { { {} }, { {} } }, i64 } %0, { i1, {} } zeroinitializer, 0, !dbg !8
  store { { i1, {} }, ptr, { { {} }, { {} } }, i64 } %1, ptr %tmp, align 8
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_selectiondag.cpp", directory: "/home/AnonTokyo/documents/llvm-project/temp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DILocalVariable(name: "v1", scope: !4, file: !1, line: 2)
!6 = !DILocation(line: 2, column: 1, scope: !4)
!7 = !DILocalVariable(name: "v2", scope: !4, file: !1, line: 3)
!8 = !DILocation(line: 3, column: 1, scope: !4)
