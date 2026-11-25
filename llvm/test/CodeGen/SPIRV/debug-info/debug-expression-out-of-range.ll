; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=ONE --implicit-check-not=DebugOperation --implicit-check-not=DebugExpression --implicit-check-not=DebugDeclare
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; NonSemantic sets take no literals, so every DIExpression argument becomes a
; 32-bit OpConstant. An argument that does not fit is dropped.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[OK:%[0-9]+]] = OpString "ok"
; CHECK-DAG: [[BIG:%[0-9]+]] = OpString "big"
; CHECK-DAG: [[GBIG:%[0-9]+]] = OpString "gbig"
; The trailing anchors keep e.g. [[C4]] from binding to "OpConstant %3 40".
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C4:%[0-9]+]] = OpConstant [[I32T]] 4{{ *$}}
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32T]] 5{{ *$}}
; CHECK-DAG: [[C6:%[0-9]+]] = OpConstant [[I32T]] 6{{ *$}}
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[NONE:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInfoNone

; Both variables are collected and emitted; only the expressions differ.
; CHECK-DAG: [[OKVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[OK]]
; CHECK-DAG: [[BIGVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[BIG]]

; CHECK-DAG: [[CONSTU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C4]]{{ *$}}
; CHECK-DAG: [[SWAP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C5]]{{ *$}}
; CHECK-DAG: [[XDEREF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C6]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[CONSTU]] [[SWAP]] [[XDEREF]]{{ *$}}

; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[GBIG]] {{.*}} [[GBIG]] [[NONE]]

; Both allocas still become OpVariable, in declaration order; only the declare
; for the first one can be emitted.
; CHECK: [[OKADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK: [[BIGADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[OKVAR]] [[OKADDR]] [[EXPR]]

; ONE: OpExtInst {{.*}} DebugOperation
; ONE: OpExtInst {{.*}} DebugOperation
; ONE: OpExtInst {{.*}} DebugOperation
; ONE: OpExtInst {{.*}} DebugExpression
; ONE: OpExtInst {{.*}} DebugDeclare

target triple = "spirv64-unknown-unknown"

define spir_func i32 @f(i32 noundef %x) !dbg !10 {
entry:
  %ok = alloca i32, align 4
  %big = alloca i32, align 4
  store i32 %x, ptr %ok, align 4
    #dbg_declare(ptr %ok, !11, !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef), !13)
    #dbg_declare(ptr %big, !12, !DIExpression(DW_OP_constu, 4294967296, DW_OP_swap, DW_OP_xderef), !13)
  %0 = load i32, ptr %ok, align 4, !dbg !13
  store i32 %0, ptr %big, align 4, !dbg !13
  %1 = load i32, ptr %big, align 4, !dbg !13
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "t.c", directory: "/src")
!4 = !{!5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression(DW_OP_constu, 4294967296, DW_OP_swap, DW_OP_xderef))
!6 = distinct !DIGlobalVariable(name: "gbig", linkageName: "gbig", scope: !0, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !9)
!9 = !{!7, !7}
!10 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocalVariable(name: "ok", scope: !10, file: !3, line: 4, type: !7)
!12 = !DILocalVariable(name: "big", scope: !10, file: !3, line: 5, type: !7)
!13 = !DILocation(line: 4, column: 7, scope: !10)
