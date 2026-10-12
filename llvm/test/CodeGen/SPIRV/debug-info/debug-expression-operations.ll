; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; One expression using the operations no other test reaches: Deref (0),
; Plus (1), Minus (2), PlusUconst (3), StackValue (7) and Fragment (9).

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C1:%[0-9]+]] = OpConstant [[I32T]] 1{{ *$}}
; CHECK-DAG: [[C2:%[0-9]+]] = OpConstant [[I32T]] 2{{ *$}}
; CHECK-DAG: [[C3:%[0-9]+]] = OpConstant [[I32T]] 3{{ *$}}
; CHECK-DAG: [[C4:%[0-9]+]] = OpConstant [[I32T]] 4{{ *$}}
; CHECK-DAG: [[C7:%[0-9]+]] = OpConstant [[I32T]] 7{{ *$}}
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[C9:%[0-9]+]] = OpConstant [[I32T]] 9{{ *$}}

; CHECK-DAG: [[DEREF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C0]]{{ *$}}
; CHECK-DAG: [[PLUSU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C3]] [[C4]]{{ *$}}
; CHECK-DAG: [[CONSTU8:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C8]]{{ *$}}
; CHECK-DAG: [[MINUS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C2]]{{ *$}}
; CHECK-DAG: [[CONSTU2:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C2]]{{ *$}}
; CHECK-DAG: [[PLUS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C1]]{{ *$}}
; CHECK-DAG: [[STACK:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C7]]{{ *$}}
; CHECK-DAG: [[FRAG:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C9]] [[C0]] [[C8]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[DEREF]] [[PLUSU]] [[CONSTU8]] [[MINUS]] [[CONSTU2]] [[PLUS]] [[STACK]] [[FRAG]]{{ *$}}

; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPR]]

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
    #dbg_declare(ptr %x, !9, !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4, DW_OP_constu, 8, DW_OP_minus, DW_OP_constu, 2, DW_OP_plus, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 8), !10)
  ; 99 keeps this constant clear of the operation encodings captured above.
  store i32 99, ptr %x, align 4, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-operations.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 1, scope: !5)
