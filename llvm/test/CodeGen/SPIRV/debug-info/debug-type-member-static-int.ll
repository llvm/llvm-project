; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; struct S { static int inst; }; int S::inst;
; The declaration is the only link to S, so S must be emitted and must list inst.

; CHECK-DAG: [[M:%[0-9]+]] = {{.*}} DebugTypeMember
; CHECK-DAG: DebugTypeComposite {{.*}} [[M]]
; CHECK-DAG: DebugGlobalVariable {{.*}} [[M]]

target triple = "spirv64-unknown-unknown"

@inst = addrspace(1) global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "inst", scope: !2, file: !4, type: !5, isDefinition: true, declaration: !6)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, emissionKind: FullDebug, globals: !7)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIFile(filename: "t.cpp", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "inst", scope: !8, file: !4, baseType: !5, flags: DIFlagStaticMember)
!7 = !{!0}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !4, size: 8, elements: !9)
!9 = !{!6}
