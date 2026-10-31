; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=COUNT --implicit-check-not=DebugOperation --implicit-check-not=DebugExpression
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Two distinct DIExpression nodes share Deref and PlusUconst 4 in opposite
; order. They cannot share a DebugExpression (different MDNodes, different
; operand order), but each DebugOperation must be emitted once and reused.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; The trailing anchors keep e.g. [[C4]] from binding to "OpConstant %3 40".
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C3:%[0-9]+]] = OpConstant [[I32T]] 3{{ *$}}
; CHECK-DAG: [[C4:%[0-9]+]] = OpConstant [[I32T]] 4{{ *$}}

; CHECK-DAG: [[ANAME:%[0-9]+]] = OpString "a"
; CHECK-DAG: [[BNAME:%[0-9]+]] = OpString "b"
; CHECK-DAG: [[DEREF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C0]]{{ *$}}
; CHECK-DAG: [[PLUSU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C3]] [[C4]]{{ *$}}
; CHECK-DAG: [[EXPR1:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[PLUSU]] [[DEREF]]{{ *$}}
; CHECK-DAG: [[EXPR2:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[DEREF]] [[PLUSU]]{{ *$}}

; CHECK-DAG: [[AVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[ANAME]]
; CHECK-DAG: [[BVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[BNAME]]

; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[AVAR]] {{%[0-9]+}} [[EXPR1]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[BVAR]] {{%[0-9]+}} [[EXPR2]]

; COUNT: OpExtInst {{.*}} DebugOperation
; COUNT: OpExtInst {{.*}} DebugOperation
; COUNT: OpExtInst {{.*}} DebugExpression
; COUNT: OpExtInst {{.*}} DebugExpression

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
    #dbg_declare(ptr %a, !9, !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref), !11)
    #dbg_declare(ptr %b, !10, !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4), !11)
  store i32 0, ptr %a, align 4, !dbg !11
  store i32 0, ptr %b, align 4, !dbg !11
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-operation-cache.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "a", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocalVariable(name: "b", scope: !5, file: !1, line: 3, type: !7)
!11 = !DILocation(line: 4, column: 1, scope: !5)
