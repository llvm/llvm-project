; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; External call appears before the alloca in LLVM IR source order. The local
; OpVariable is still inserted in the function preamble; DebugFunctionDefinition
; must follow it, not the entry OpLabel.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "caller"
; CHECK-DAG: [[NAME_NOARGS:%[0-9]+]] = OpString "caller_no_args"
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME]]
; CHECK-DAG: [[DF_NOARGS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_NOARGS]]
; CHECK-DAG: OpName [[EXT_HELPER:%[0-9]+]] "external_helper"
; CHECK-DAG: OpDecorate [[EXT_HELPER]] LinkageAttributes "external_helper" Import
; CHECK-DAG: OpDecorate [[CALLER:%[0-9]+]] LinkageAttributes "caller" Export
; CHECK-DAG: OpName [[EXT_HELPER_NOARGS:%[0-9]+]] "external_helper_no_args"
; CHECK-DAG: OpDecorate [[EXT_HELPER_NOARGS]] LinkageAttributes "external_helper_no_args" Import
; CHECK-DAG: OpDecorate [[CALLER_NOARGS:%[0-9]+]] LinkageAttributes "caller_no_args" Export

; external_helper: hoisted declaration in the declarations section.
; CHECK: [[EXT_HELPER]] = OpFunction %{{.*}}
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpFunctionEnd

; CHECK: [[EXT_HELPER_NOARGS]] = OpFunction %{{.*}}
; CHECK-NEXT: OpFunctionEnd

; CHECK: [[CALLER]] = OpFunction %{{.*}} ; -- Begin function caller
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpVariable {{.*}} Function
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[CALLER]]
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

; caller_no_args: same placement rules, but no function parameters.
; CHECK: [[CALLER_NOARGS]] = OpFunction %{{.*}} ; -- Begin function caller_no_args
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpVariable {{.*}} Function
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_NOARGS]] [[CALLER_NOARGS]]
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

declare spir_func i32 @external_helper(i32)
declare spir_func i32 @external_helper_no_args()

define spir_func i32 @caller(i32 %x) !dbg !5 {
entry:
  %r = call i32 @external_helper(i32 %x)
  %a = alloca i32, align 4
  store i32 %r, ptr %a
  ret i32 %r, !dbg !8
}

define spir_func void @caller_no_args() !dbg !9 {
entry:
  %r = call i32 @external_helper_no_args()
  %a = alloca i32, align 4
  store i32 %r, ptr %a
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-function-definition-call-before-alloca.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 6, column: 3, scope: !5)

!9 = distinct !DISubprogram(name: "caller_no_args", linkageName: "caller_no_args", scope: !1, file: !1, line: 10, type: !10, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!13 = !DILocation(line: 15, column: 3, scope: !9)
