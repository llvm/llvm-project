; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that a DINamespace becomes a DebugLexicalBlock when it is the Parent of
; a DebugGlobalVariable. Two cases are covered: a named namespace nested in
; another named one, and an anonymous namespace nested in a named one.
; Per the NSDI spec, an anonymous namespace must get an empty OpString as its
; Name operand. Clang emits such namespaces as `!DINamespace(scope: ...)` with
; no `name:` field, so DINamespace::getName() returns "".

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[OUTER_NAME:%[0-9]+]] = OpString "outer"
; CHECK-DAG: [[INNER_NAME:%[0-9]+]] = OpString "inner"
; CHECK-DAG: [[GNAME:%[0-9]+]] = OpString "g"
; CHECK-DAG: [[HNAME:%[0-9]+]] = OpString "h"
; CHECK-DAG: [[EMPTY:%[0-9]+]] = OpString ""
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit {{.*}}
; CHECK-DAG: [[OUTER_LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} [[CU]] [[OUTER_NAME]]
; CHECK-DAG: [[INNER_LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} [[OUTER_LB]] [[INNER_NAME]]
; CHECK-DAG: [[ANON_LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} [[OUTER_LB]] [[EMPTY]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[GNAME]] {{.*}} [[INNER_LB]] {{.*}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[HNAME]] {{.*}} [[ANON_LB]] {{.*}}

target triple = "spirv64-unknown-unknown"

@g = addrspace(1) global i32 0, !dbg !13
@h = addrspace(1) global i32 0, !dbg !16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !12, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-lexical-block-namespace.cpp", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DINamespace(name: "outer", scope: !1)
!9 = !DINamespace(name: "inner", scope: !8)
!10 = !DINamespace(scope: !8)

!12 = !{!13, !16}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "g", linkageName: "g", scope: !9, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "h", linkageName: "h", scope: !10, file: !1, line: 5, type: !7, isLocal: true, isDefinition: true)
