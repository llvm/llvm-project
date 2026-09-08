; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DIGlobalVariable with file: null (verifier-legal). Empty full path is lowered via
; OpString "", DebugSource for that file operand, then DebugGlobalVariable.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}tmp{{[/\\]}}path-null.c"
; CHECK-DAG: [[EMPTY:%[0-9]+]] = OpString ""
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "no_file_path"
; CHECK-DAG: [[STR_INT:%[0-9]+]] = OpString "int"
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32T]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32T]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32T]] 42
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8
; CHECK-DAG: [[DS_CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS_CU]] [[C0]]
; CHECK-DAG: [[DTI:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[STR_INT]]
; CHECK-DAG: [[DS_GV:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[EMPTY]]
; CHECK-DAG: [[GV:%[0-9]+]] = OpVariable {{.*}} CrossWorkgroup
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAME]] [[DTI]] [[DS_GV]] [[C42]] [[C0]] [[CU]] [[NAME]] [[GV]] [[C8]]

target triple = "spirv64-unknown-unknown"

@no_file_path = dso_local addrspace(1) global i32 0, align 4, !dbg !0

define spir_func void @f() !dbg !9 {
entry:
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "no_file_path", linkageName: "no_file_path", scope: !2, file: null, line: 42, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "path-null.c", directory: "/tmp")
!4 = !{!0}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!10 = !DILocation(line: 2, column: 1, scope: !9)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !17)
!17 = !{null}
