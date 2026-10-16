; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise NonSemantic DebugFunctionDefinition for defined functions, including
; per-function emitter state across multiple definitions in one module.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-function-definition.c"
; CHECK-DAG: [[NAME1:%[0-9]+]] = OpString "add_one"
; CHECK-DAG: [[NAME2:%[0-9]+]] = OpString "add_two"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit {{.*}}
; CHECK-DAG: [[TF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeFunction {{.*}}
; CHECK-DAG: [[DF1:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME1]] [[TF]] [[DS]] {{.*}}
; CHECK-DAG: [[DF2:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME2]] [[TF]] [[DS]] {{.*}}
; CHECK: [[ADD_ONE:%[0-9]+]] = OpFunction
; CHECK: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF1]] [[ADD_ONE]]
; CHECK: [[ADD_TWO:%[0-9]+]] = OpFunction
; CHECK: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF2]] [[ADD_TWO]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_one(i32 %value) !dbg !5 {
entry:
  %result = add i32 %value, 1
  ret i32 %result, !dbg !8
}

define spir_func i32 @add_two(i32 %value) !dbg !9 {
entry:
  %result = add i32 %value, 2
  ret i32 %result, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-function-definition.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 3, scope: !5)

!9 = distinct !DISubprogram(name: "add_two", linkageName: "add_two", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 7, column: 3, scope: !9)
