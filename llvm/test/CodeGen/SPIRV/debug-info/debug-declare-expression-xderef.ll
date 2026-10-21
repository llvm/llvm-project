; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The address-space expression clang puts on every declare for a SPIR-V target.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[X:%[0-9]+]] = OpString "x"
; CHECK-DAG: [[Y:%[0-9]+]] = OpString "y"
; The trailing anchors keep e.g. [[C8]] from binding to "OpConstant %3 80".
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32T]] 5{{ *$}}
; CHECK-DAG: [[C6:%[0-9]+]] = OpConstant [[I32T]] 6{{ *$}}
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[XVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[X]]
; CHECK-DAG: [[YVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[Y]]

; CHECK-DAG: [[CONSTU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C8]] [[C0]]{{ *$}}
; CHECK-DAG: [[SWAP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C5]]{{ *$}}
; CHECK-DAG: [[XDEREF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C6]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[CONSTU]] [[SWAP]] [[XDEREF]]{{ *$}}

; CHECK: [[XADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK: [[YADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[XVAR]] [[XADDR]] [[EXPR]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[YVAR]] [[YADDR]] [[EXPR]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @sum(i32 noundef %x) !dbg !5 {
entry:
  %x.addr = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
    #dbg_declare(ptr %x.addr, !9, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !11)
    #dbg_declare(ptr %y, !10, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !12)
  %0 = load i32, ptr %x.addr, align 4, !dbg !12
  store i32 %0, ptr %y, align 4, !dbg !12
  %1 = load i32, ptr %y, align 4, !dbg !12
  ret i32 %1, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-expression-xderef.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "sum", linkageName: "sum", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 1, type: !7)
!10 = !DILocalVariable(name: "y", scope: !5, file: !1, line: 2, type: !7)
!11 = !DILocation(line: 1, column: 13, scope: !5)
!12 = !DILocation(line: 2, column: 7, scope: !5)
