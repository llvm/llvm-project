; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=COUNT --implicit-check-not=DebugOperation --implicit-check-not=DebugExpression
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VAL %}

; A #dbg_declare and a DIGlobalVariableExpression share DW_OP_constu 42,
; DW_OP_stack_value. There is no llvm::GlobalVariable: a backing @g would
; make DebugGlobalVariable use that OpVariable and skip the expression.
; We check that both reuse one DebugExpression.
;
; spirv-val rejects a DebugExpression as the DebugGlobalVariable Variable
; operand (KhronosGroup/SPIRV-Tools#6469). The VAL line expects that failure;
; when the fix lands, restore a plain spirv-val invocation.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[XNAME:%[0-9]+]] = OpString "x"
; CHECK-DAG: [[SMALL:%[0-9]+]] = OpString "Small"
; The trailing anchors keep e.g. [[C7]] from binding to "OpConstant %3 72".
; CHECK-DAG: [[C7:%[0-9]+]] = OpConstant [[I32T]] 7{{ *$}}
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32T]] 42{{ *$}}

; CHECK-DAG: [[CONSTU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C42]]{{ *$}}
; CHECK-DAG: [[SV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C7]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[CONSTU]] [[SV]]{{ *$}}

; CHECK-DAG: [[XVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[XNAME]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[SMALL]] {{.*}} [[SMALL]] [[EXPR]]

; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[XVAR]] {{%[0-9]+}} [[EXPR]]

; COUNT: OpExtInst {{.*}} DebugOperation
; COUNT: OpExtInst {{.*}} DebugOperation
; COUNT: OpExtInst {{.*}} DebugExpression

; The operand list the validator will accept has varied across releases, so
; match only up to it.
; VAL: DebugGlobalVariable: expected operand Variable must be a result id of

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
    #dbg_declare(ptr %x, !9, !DIExpression(DW_OP_constu, 42, DW_OP_stack_value), !10)
  store i32 0, ptr %x, align 4, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !11, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-cache.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !{!12}
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression(DW_OP_constu, 42, DW_OP_stack_value))
!13 = distinct !DIGlobalVariable(name: "Small", linkageName: "Small", scope: !0, file: !1, line: 4, type: !7, isLocal: true, isDefinition: true)
