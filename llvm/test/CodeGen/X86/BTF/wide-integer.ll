; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s --implicit-check-not='BTF_KIND_INT' --implicit-check-not='BTF_KIND_STRUCT'
; CHECK: BTF_KIND_FLOAT

@w = global i256 0, !dbg !8
@wide = global i256 0, !dbg !0
@ok = global float 0.0, !dbg !11
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3, !4}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "wide", scope: !2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !6, emissionKind: FullDebug, globals: !{!8, !0, !11})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"BTF", i32 1}
!5 = !DIBasicType(name: "unsigned _BitInt(256)", size: 256, encoding: DW_ATE_unsigned)
!6 = !DIFile(filename: "test.c", directory: "/")
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Wrapper", size: 256, elements: !{!10})
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "w", scope: !2, type: !7, isLocal: false, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !7, baseType: !5, size: 256)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "ok", scope: !2, type: !13, isLocal: false, isDefinition: true)
!13 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
