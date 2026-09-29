; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o %t.spt
; RUN: FileCheck %s --check-prefix=CHECK --input-file %t.spt
; RUN: FileCheck %s --check-prefix=COUNT --input-file %t.spt
; RUN: FileCheck %s --check-prefix=NO-COMPOSITE --input-file %t.spt
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DINamespace parented by a function-body DILexicalBlock. Clang never emits this for C++;
; the IR verifier allows it. Parent-before-child order holds within each kind, not
; across kinds, so the namespace block and composite are skipped.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[FNNAME:%[0-9]+]] = OpString "fn"
; CHECK-DAG: [[MNAME:%[0-9]+]] = OpString "m"
; CHECK-DAG: OpString "weird"
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[FNNAME]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} [[DF]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugTypeMember [[MNAME]]

; COUNT-COUNT-1: DebugLexicalBlock

; NO-COMPOSITE-NOT: DebugTypeComposite

target triple = "spirv64-unknown-unknown"

define spir_func void @fn() !dbg !5 {
entry:
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !20, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "namespace-in-block.cpp", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "fn", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!10 = distinct !DILexicalBlock(scope: !5, file: !1, line: 2, column: 3)
!11 = !DILocation(line: 2, column: 3, scope: !10)
!8 = !DINamespace(name: "weird", scope: !10)

!20 = !{!21}
!21 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !8, file: !1, line: 4, size: 32, elements: !23, identifier: "_ZTS1S")
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !21, file: !1, line: 5, baseType: !7, size: 32)
