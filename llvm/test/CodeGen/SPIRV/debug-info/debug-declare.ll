; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A #dbg_declare on an alloca that survives to MIR: the parameter copy and a
; local. Both use an empty DIExpression, so a single DebugExpression with no
; operations is shared by both declares.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[VALUE:%[0-9]+]] = OpString "value"
; CHECK-DAG: [[RESULT:%[0-9]+]] = OpString "result"
; CHECK-DAG: [[VALUEVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[VALUE]]
; CHECK-DAG: [[RESULTVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[RESULT]]
; An empty DIExpression lowers to a DebugExpression with no operands.
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}

; CHECK: [[VALUEADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK: [[RESULTADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[VALUEVAR]] [[VALUEADDR]] [[EXPR]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[RESULTVAR]] [[RESULTADDR]] [[EXPR]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_one(i32 %value) !dbg !5 {
entry:
  %value.addr = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
    #dbg_declare(ptr %value.addr, !9, !DIExpression(), !11)
  %0 = load i32, ptr %value.addr, align 4, !dbg !11
  %add = add nsw i32 %0, %0, !dbg !11
  store i32 %add, ptr %result, align 4, !dbg !11
    #dbg_declare(ptr %result, !10, !DIExpression(), !12)
  %1 = load i32, ptr %result, align 4, !dbg !12
  ret i32 %1, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 7, type: !4, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "value", arg: 1, scope: !5, file: !1, line: 7, type: !7)
!10 = !DILocalVariable(name: "result", scope: !5, file: !1, line: 11, type: !7)
!11 = !DILocation(line: 7, column: 20, scope: !5)
!12 = !DILocation(line: 11, column: 30, scope: !5)
