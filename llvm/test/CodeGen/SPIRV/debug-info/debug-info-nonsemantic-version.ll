; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-100
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info -spirv-nonsemantic-debug-info-version=200 %s -o - | FileCheck %s --check-prefix=CHECK-200
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -spirv-nonsemantic-debug-info-version=200 -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise -spirv-nonsemantic-debug-info-version: it selects which
; NonSemantic.Shader.DebugInfo ext-inst-set name is imported (.100 by
; default, .200 when requested). 

; Check that mnemonic are properly printed and UNKNOWN_EXT_INST
; is not emitted.

; CHECK-100: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-100-DAG: %[[#I32T100:]] = OpTypeInt 32 0
; CHECK-100-DAG: %[[#C100:]] = OpConstant %[[#I32T100]] 100{{ *$}}
; CHECK-100: OpExtInst %{{.*}} %{{.*}} DebugSource
; CHECK-100: OpExtInst {{.*}} DebugCompilationUnit %[[#C100]]

; CHECK-200: OpExtInstImport "NonSemantic.Shader.DebugInfo.200"
; CHECK-200-DAG: %[[#I32T200:]] = OpTypeInt 32 0
; CHECK-200-DAG: %[[#C200:]] = OpConstant %[[#I32T200]] 200{{ *$}}
; CHECK-200: OpExtInst %{{.*}} %{{.*}} DebugSource
; CHECK-200: OpExtInst {{.*}} DebugCompilationUnit %[[#C200]]
; CHECK-200-NOT: UNKNOWN_EXT_INST

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_one(i32 %value) !dbg !5 {
entry:
  %result = add i32 %value, 1
  ret i32 %result, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-info-nonsemantic-version.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 3, scope: !5)
