; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DebugFunction should reference the emitted DebugFunctionDeclaration when the
; defining DISubprogram carries a declaration operand.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "add_one"
; CHECK-DAG: [[DECL:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunctionDeclaration [[NAME]] {{.*}} [[NAME]] {{.*}}
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME]] {{.*}} [[DECL]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_one(i32 %value) !dbg !5 {
entry:
  %result = add i32 %value, 1
  ret i32 %result, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, retainedTypes: !9)
!1 = !DIFile(filename: "debug-function-with-declaration.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!9 = !{!10}
!10 = !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: 0)

!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !10)
!8 = !DILocation(line: 3, column: 3, scope: !5)
