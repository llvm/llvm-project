; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise NonSemantic DebugGlobalVariable for a module global with !dbg.
; The global uses address space 1 so the backend emits a module-scope
; CrossWorkgroup OpVariable (default-AS globals become function-local copies).

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}AAAAAAAAAA{{[/\\]}}BBBBBBBB{{[/\\]}}CCCCCCCCC{{[/\\]}}debug-global-variable.c"
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "g_value"
; CHECK-DAG: [[STR_INT:%[0-9]+]] = OpString "int"
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32T]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32T]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32T]] 42
; CHECK-DAG: [[C8:%[0-9]+]] = OpConstant [[I32T]] 8
; CHECK-DAG: [[C32:%[0-9]+]] = OpConstant [[I32T]] 32
; CHECK-DAG: [[C4ENC:%[0-9]+]] = OpConstant [[I32T]] 4
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS]] [[C0]]
; CHECK-DAG: [[DTI:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[STR_INT]] [[C32]] [[C4ENC]] [[C0]]
; CHECK-DAG: [[GV:%[0-9]+]] = OpVariable {{.*}} CrossWorkgroup
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAME]] [[DTI]] [[DS]] [[C42]] [[C0]] [[CU]] [[NAME]] [[GV]] [[C8]]

target triple = "spirv64-unknown-unknown"

@g_value = dso_local addrspace(1) global i32 0, align 4, !dbg !0

define spir_func void @use_global() !dbg !9 {
entry:
  %v = load i32, ptr addrspace(1) @g_value, align 4, !dbg !10
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_value", linkageName: "g_value", scope: !2, file: !3, line: 42, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "debug-global-variable.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!4 = !{!0}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "use_global", linkageName: "use_global", scope: !3, file: !3, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!10 = !DILocation(line: 2, column: 3, scope: !9)
!11 = !DILocation(line: 3, column: 1, scope: !9)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !17)
!17 = !{!8}
