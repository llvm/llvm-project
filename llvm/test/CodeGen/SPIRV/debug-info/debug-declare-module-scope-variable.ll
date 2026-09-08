; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A #dbg_declare whose address is a module-scope global rather than an alloca.
; The G_GLOBAL_VALUE is selected to the Workgroup OpVariable, so the storage
; operand resolves and the declare is emitted from inside the function.
;
; This is a synthetic case, clang doesn't seem to emit this.
; clang describes an OpenCL __local or a HIP __shared__ variable with a
; DIGlobalVariable, which takes the DebugGlobalVariable path instead.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "shared"
; CHECK-DAG: [[STORAGE:%[0-9]+]] = OpVariable {{%[0-9]+}} Workgroup
; CHECK-DAG: [[VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[NAME]]
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}

; CHECK: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[VAR]] [[STORAGE]] [[EXPR]]

target triple = "spirv64-unknown-unknown"

@shared = internal addrspace(3) global i32 undef, align 4

define spir_func void @k() !dbg !5 {
entry:
    #dbg_declare(ptr addrspace(3) @shared, !9, !DIExpression(), !10)
  store i32 1, ptr addrspace(3) @shared, align 4, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-module-scope-variable.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "k", linkageName: "k", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "shared", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 3, scope: !5)
