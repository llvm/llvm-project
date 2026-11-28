; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --implicit-check-not=DebugDeclare
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_untyped_pointers --spirv-debug-declare-untyped-variable=true %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ON
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_untyped_pointers --spirv-debug-declare-untyped-variable=true %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VAL-ON %}

; With untyped pointers a local's storage is an OpUntypedVariableKHR.
; Naming it from DebugDeclare is not allowed in NonSemantic.Shader.DebugInfo.
; It requires the Variable operand to be an OpVariable.
; -spirv-debug-declare-untyped-variable relaxes this.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[X:%[0-9]+]] = OpString "x"
; CHECK-DAG: [[LV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[X]]
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}
; CHECK: [[VAR:%[0-9]+]] = OpUntypedVariableKHR {{%[0-9]+}} Function
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition
; CHECK-ON: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[LV]] [[VAR]] [[EXPR]]

; VAL-ON: error: {{.*}}NonSemantic.Shader.DebugInfo.100 DebugDeclare: expected operand Variable must be a result id of OpVariable or OpFunctionParameter

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
    #dbg_declare(ptr %x, !9, !DIExpression(), !10)
  store i32 1, ptr %x, align 4, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-untyped-variable.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 1, scope: !5)
