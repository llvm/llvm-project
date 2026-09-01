; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NS:%[0-9]+]] = OpString "outer"
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock {{.*}} [[CU]] [[NS]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypeComposite {{.*}} [[LB]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypedef {{.*}} [[CU]]

target triple = "spirv64-unknown-unknown"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "t.cpp", directory: "/src")
!4 = !{!5, !6}
!7 = !DINamespace(name: "outer", scope: !3)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !7, file: !3, line: 1, size: 32, elements: !10)
!10 = !{}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", scope: !7, file: !3, line: 2, baseType: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
