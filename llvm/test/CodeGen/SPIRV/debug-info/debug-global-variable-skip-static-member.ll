; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A DIGlobalVariable that is the definition of a static data member. Its
; declaration DIDerivedType (DW_TAG_member) is not emitted into DebugTypeRegs
; (member types are not supported yet), so the Static Member Declaration operand
; cannot be resolved and the whole DebugGlobalVariable is skipped.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: OpExtInst {{.*}} DebugCompilationUnit
; CHECK-NOT: DebugGlobalVariable

target triple = "spirv64-unknown-unknown"

@g = dso_local addrspace(1) global i32 0, align 4, !dbg !0

define spir_func void @f() !dbg !9 {
entry:
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "member", linkageName: "member", scope: !2, file: !3, line: 42, type: !8, isLocal: false, isDefinition: true, declaration: !20)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.cpp", directory: "/tmp")
!4 = !{!0}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!10 = !DILocation(line: 2, column: 1, scope: !9)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !17)
!17 = !{null}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "member", scope: !21, file: !3, line: 2, baseType: !8, flags: DIFlagStaticMember)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 8, elements: !22, identifier: "_ZTS1A")
!22 = !{!20}
