; RUN: llc -O1 < %s
; REQUIRES: default_triple

; ModuleID = 'pr6157.bc'
; formerly crashed in SelectionDAGBuilder

%tart.reflect.ComplexType = type { double, double }

@.type.SwitchStmtTest = constant %tart.reflect.ComplexType { double 3.0, double 2.0 }

define i32 @"main(tart.core.String[])->int32"(i32 %args) {
entry:
  tail call void @llvm.dbg.value(metadata %tart.reflect.ComplexType* @.type.SwitchStmtTest, i64 0, metadata !8, metadata !DIExpression()), !dbg !DILocation(scope: !9)
  tail call void @"tart.reflect.ComplexType.create->tart.core.Object"(%tart.reflect.ComplexType* @.type.SwitchStmtTest) ; <%tart.core.Object*> [#uses=2]
  ret i32 3
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone
declare void @"tart.reflect.ComplexType.create->tart.core.Object"(%tart.reflect.ComplexType*) nounwind readnone

!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !15, enums: !16, retainedTypes: !16)
!1 = !DIDerivedType(tag: DW_TAG_const_type, size: 192, align: 64, file: !15, scope: !0, baseType: !2)
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", line: 1, size: 192, align: 64, file: !15, scope: !0, elements: !3)
!3 = !{!4, !6, !7}
!4 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 1, size: 64, align: 64, file: !15, scope: !2, baseType: !5)
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 1, size: 64, align: 64, offset: 64, file: !15, scope: !2, baseType: !5)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "z", line: 1, size: 64, align: 64, offset: 128, file: !15, scope: !2, baseType: !5)
!8 = !DILocalVariable(name: "t", line: 5, scope: !9, file: !0, type: !2)
!9 = distinct !DILexicalBlock(line: 0, column: 0, file: null, scope: !10)
!10 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scope: !0, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{%tart.reflect.ComplexType* @.type.SwitchStmtTest}
!15 = !DIFile(filename: "sm.c", directory: "")
!16 = !{i32 0}
