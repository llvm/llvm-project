; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A function parameter and a local variable.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-local-variable.c"
; CHECK-DAG: [[FNNAME:%[0-9]+]] = OpString "add_one"
; CHECK-DAG: [[INTNAME:%[0-9]+]] = OpString "int"
; CHECK-DAG: [[VALUE:%[0-9]+]] = OpString "value"
; CHECK-DAG: [[RESULT:%[0-9]+]] = OpString "result"
; The trailing anchors keep e.g. [[C1]] from binding to "OpConstant %3 100".
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C1:%[0-9]+]] = OpConstant [[I32T]] 1{{ *$}}
; CHECK-DAG: [[C7:%[0-9]+]] = OpConstant [[I32T]] 7{{ *$}}
; CHECK-DAG: [[C11:%[0-9]+]] = OpConstant [[I32T]] 11{{ *$}}
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[INT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[INTNAME]] {{.*}} [[C0]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[FNNAME]] {{.*}} [[DS]] {{.*}}

; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[VALUE]] [[INT]] [[DS]] [[C7]] [[C0]] [[DF]] [[C0]] [[C1]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[RESULT]] [[INT]] [[DS]] [[C11]] [[C0]] [[DF]] [[C0]]

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
!1 = !DIFile(filename: "debug-local-variable.c", directory: "/src")
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
