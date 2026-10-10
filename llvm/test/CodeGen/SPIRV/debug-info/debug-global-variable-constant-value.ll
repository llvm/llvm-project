; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=ONE --implicit-check-not=DebugOperation --implicit-check-not=DebugExpression
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VAL %}

; This HIP code compiles to the metadata below with no optimisations as it is the case for amdgcnspirv:
;
;   constexpr unsigned long long Hash = 0xff51afd7ed558ccdULL;
;   constexpr double Scale = 1.5;
;   constexpr unsigned Small = 42u;
;

; DebugOperation requires int32 operands and DI for Hash and Scale cannot be encoded.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[HASH:%[0-9]+]] = OpString "Hash"
; CHECK-DAG: [[SCALE:%[0-9]+]] = OpString "Scale"
; CHECK-DAG: [[SMALL:%[0-9]+]] = OpString "Small"
; The trailing anchors keep e.g. [[C7]] from binding to "OpConstant %3 72".
; CHECK-DAG: [[C7:%[0-9]+]] = OpConstant [[I32T]] 7{{ *$}}
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32T]] 42{{ *$}}
; CHECK-DAG: [[NONE:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInfoNone

; CHECK-DAG: [[CONSTU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C42]]{{ *$}}
; CHECK-DAG: [[SV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C7]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[CONSTU]] [[SV]]{{ *$}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[SMALL]] {{.*}} [[SMALL]] [[EXPR]]

; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[HASH]] {{.*}} [[HASH]] [[NONE]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[SCALE]] {{.*}} [[SCALE]] [[NONE]]

; Small's two operations and one expression are the only ones emitted.
; ONE: OpExtInst {{.*}} DebugOperation
; ONE: OpExtInst {{.*}} DebugOperation
; ONE: OpExtInst {{.*}} DebugExpression

; spirv-val is broken and is rejecting an expression in a DebugGlobalVariable (KhronosGroup/SPIRV-Tools#6469).
; VAL: DebugGlobalVariable: expected operand Variable must be a result id of

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !20 {
entry:
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HIP, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "fe.hip", directory: "/src")
!4 = !{!5, !8, !11}

; constexpr unsigned long long Hash = 0xff51afd7ed558ccdULL;
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression(DW_OP_constu, 18397679294719823053, DW_OP_stack_value))
!6 = distinct !DIGlobalVariable(name: "Hash", linkageName: "Hash", scope: !0, file: !3, line: 2, type: !7, isLocal: true, isDefinition: true)
!7 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)

; constexpr double Scale = 1.5;  (0x3FF8000000000000)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression(DW_OP_constu, 4609434218613702656, DW_OP_stack_value))
!9 = distinct !DIGlobalVariable(name: "Scale", linkageName: "Scale", scope: !0, file: !3, line: 3, type: !10, isLocal: true, isDefinition: true)
!10 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)

; constexpr unsigned Small = 42u;
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression(DW_OP_constu, 42, DW_OP_stack_value))
!12 = distinct !DIGlobalVariable(name: "Small", linkageName: "Small", scope: !0, file: !3, line: 4, type: !13, isLocal: true, isDefinition: true)
!13 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)

!18 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !19)
!19 = !{null}
!20 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 6, type: !18, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocation(line: 7, column: 1, scope: !20)
