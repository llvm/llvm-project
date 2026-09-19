; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info -spirv-nonsemantic-debug-info-version=200 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown -spirv-nonsemantic-debug-info-version=200 %s -o - -filetype=obj | spirv-val %}

; Extra debug operations (encoding > Fragment) are only emitted when the
; NonSemantic.Shader.DebugInfo.200 extension set is selected.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.200"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[C10:%[0-9]+]] = OpConstant [[I32T]] 10{{ *$}}
; CHECK-DAG: [[C30:%[0-9]+]] = OpConstant [[I32T]] 30{{ *$}}
; CHECK-DAG: [[C32:%[0-9]+]] = OpConstant [[I32T]] 32{{ *$}}
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32T]] 5{{ *$}}

; CHECK-DAG: [[CONV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C10]] [[C32]] [[C5]]{{ *$}}
; CHECK-DAG: [[MUL:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[C30]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[CONV]] [[MUL]]{{ *$}}
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPR]]

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
    #dbg_declare(ptr %x, !9, !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_mul), !10)
  store i32 1, ptr %x, align 4, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-extra-ops-200.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 1, scope: !5)
