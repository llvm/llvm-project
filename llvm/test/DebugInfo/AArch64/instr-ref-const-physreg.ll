; RUN: llc %s -stop-before=finalize-isel -mtriple=aarch64 -o - \
; RUN:     -experimental-debug-variable-locations | FileCheck %s

; Test that when an SSA Value becomes a constant-physreg copy, under the
; instruction referencing model, the COPY is labelled. In the general case
; labelling a copy is undesirable -- this test really checks that we don't
; crash, and we don't just drop the information.

; CHECK: DBG_PHI $xzr, 1
; CHECK: DBG_INSTR_REF {{.+}}, dbg-instr-ref(1, 0)

define i64 @test() !dbg !7 {
  %foo = add i64 0, 0
  call void @llvm.dbg.value(metadata i64 %foo, metadata !12, metadata !DIExpression()), !dbg !13
  ret i64 %foo, !dbg !13
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/out.c")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "bar", arg: 1, scope: !7, file: !1, line: 3, type: !11)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocalVariable(name: "baz", arg: 2, scope: !7, file: !1, line: 3, type: !11)
