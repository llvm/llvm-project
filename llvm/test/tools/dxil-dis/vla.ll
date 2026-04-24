; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: !DISubrange(count: -1)

define void @vla(i32 noundef %s) !dbg !4 {
entry:
    #dbg_value(i32 %s, !9, !DIExpression(), !17)
    #dbg_value(i32 %s, !10, !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_stack_value), !17)
    #dbg_value(i32 0, !16, !DIExpression(), !17)
    #dbg_value(i32 poison, !16, !DIExpression(), !17)
    #dbg_value(i32 poison, !12, !DIExpression(), !17)
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "vla.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "vla", scope: !1, file: !1, line: 3, type: !5, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8, keyInstructions: true)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10, !12, !16}
!9 = !DILocalVariable(name: "s", arg: 1, scope: !4, file: !1, line: 3, type: !7)
!10 = !DILocalVariable(name: "__vla_expr0", scope: !4, type: !11, flags: DIFlagArtificial)
!11 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!12 = !DILocalVariable(name: "vla", scope: !4, file: !1, line: 10, type: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !14)
!14 = !{!15}
!15 = !DISubrange(count: !10)
!16 = !DILocalVariable(name: "i", scope: !4, file: !1, line: 11, type: !7)
!17 = !DILocation(line: 0, scope: !4)
!18 = !DILocation(line: 15, column: 1, scope: !4, atomGroup: 7, atomRank: 1)
