; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A DIGlobalVariable with no backing llvm::GlobalVariable but whose
; DIGlobalVariableExpression carries a non-empty DIExpression (a constant
; initializer). Both operations map, so the Variable operand is the resulting
; DebugExpression rather than DebugInfoNone: Constu (8) carrying 42, then
; StackValue (7) to say the operand stack holds the value itself.
; Flags encode IsLocal|IsDefinition (12).

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "constg"
; CHECK-DAG: [[STR_INT:%[0-9]+]] = OpString "int"
; The trailing anchors keep e.g. [[C7]] from binding to "OpConstant %3 72".
; CHECK-DAG: [[C7:%[0-9]+]] = OpConstant [[I32T]] 7{{ *$}}
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[C12:%[0-9]+]] = OpConstant [[I32T]] 12{{ *$}}
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32T]] 42{{ *$}}
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource
; CHECK-DAG: [[DTI:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[STR_INT]]
; CHECK-DAG: [[CONSTU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C42]]{{ *$}}
; CHECK-DAG: [[SV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C7]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[CONSTU]] [[SV]]{{ *$}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAME]] [[DTI]] [[DS]] [[C42]] {{%[0-9]+}} {{%[0-9]+}} [[NAME]] [[EXPR]] [[C12]]

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !9 {
entry:
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_constu, 42, DW_OP_stack_value))
!1 = distinct !DIGlobalVariable(name: "constg", linkageName: "constg", scope: !2, file: !3, line: 42, type: !8, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp")
!4 = !{!0}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!10 = !DILocation(line: 2, column: 1, scope: !9)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !17)
!17 = !{null}
