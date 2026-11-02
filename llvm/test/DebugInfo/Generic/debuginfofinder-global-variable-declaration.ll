; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; S is only reachable from the global's declaration: field.

; CHECK: Type: inst from /tmp/t.cpp:2 DW_TAG_member
; CHECK: Type: S from /tmp/t.cpp:1 DW_TAG_structure_type

@inst = global i32 0, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "inst", scope: !2, file: !4, line: 3, type: !5, isDefinition: true, declaration: !6)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, emissionKind: FullDebug, globals: !7)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIFile(filename: "t.cpp", directory: "/tmp")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "inst", scope: !8, file: !4, line: 2, baseType: !5, flags: DIFlagStaticMember)
!7 = !{!0}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !4, line: 1, size: 8, elements: !9)
!9 = !{!6}
