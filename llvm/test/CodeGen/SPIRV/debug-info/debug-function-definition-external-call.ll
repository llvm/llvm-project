; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Forward-declared external callee plus function-local variables: the hoisted
; OpFunction declaration must not cause DebugFunctionDefinition to be emitted
; before the last function-local OpVariable.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "caller_with_vars"
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME]]
; CHECK-DAG: OpName [[EXT_HELPER:%[0-9]+]] "external_helper"
; CHECK-DAG: OpDecorate [[EXT_HELPER]] LinkageAttributes "external_helper" Import
; CHECK-DAG: OpDecorate [[CALLER:%[0-9]+]] LinkageAttributes "caller_with_vars" Export

; external_helper: hoisted declaration in the declarations section.
; CHECK: [[EXT_HELPER]] = OpFunction %{{.*}}
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpFunctionEnd

; CHECK: [[CALLER]] = OpFunction %{{.*}} ; -- Begin function caller_with_vars
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpVariable {{.*}} Function
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[CALLER]]
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

declare spir_func i32 @external_helper(i32)

define spir_func i32 @caller_with_vars(i32 %x) !dbg !5 {
entry:
  %a = alloca i32, align 4
  store i32 %x, ptr %a
  %r = call i32 @external_helper(i32 %x)
  ret i32 %r, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-function-definition-external-call.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "caller_with_vars", linkageName: "caller_with_vars", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 5, column: 3, scope: !5)
