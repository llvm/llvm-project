; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_INTEL_unstructured_loop_controls %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_INTEL_unstructured_loop_controls -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DebugLine for the branch is emitted before OpLoopControlINTEL.
; DebugLine/DebugNoLine must not appear after it.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-line-loop-control-intel.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V50:%[0-9]+]] = OpConstant [[I32]] 50{{$}}
; CHECK-DAG: [[V51:%[0-9]+]] = OpConstant [[I32]] 51{{$}}
; CHECK-DAG: [[V99:%[0-9]+]] = OpConstant [[I32]] 99{{$}}

; Latch: DebugLine for the add, then branch's DebugLine before OpLoopControlINTEL,
; then the branch with no DebugLine/DebugNoLine in between.
; CHECK:      OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V5]] [[V5]] [[V5]] [[V6]]
; CHECK-NEXT: OpIAdd
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V99]] [[V99]] [[V50]] [[V51]]
; CHECK-NEXT: OpLoopControlINTEL Unroll
; CHECK-NEXT: OpBranch

target triple = "spirv64-unknown-unknown"

define spir_func i32 @loop(i32 %n) !dbg !5 {
entry:
  br label %header, !dbg !8

header:
  %i = phi i32 [ 0, %entry ], [ %next, %body ], !dbg !9
  %cmp = icmp slt i32 %i, %n, !dbg !10
  br i1 %cmp, label %body, label %exit, !dbg !10

body:
  %next = add i32 %i, 1, !dbg !11
  call void (...) @llvm.spv.loop.control.intel(i32 1), !dbg !14
  br label %header, !dbg !13

exit:
  ret i32 %i, !dbg !12
}

declare void @llvm.spv.loop.control.intel(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-line-loop-control-intel.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "loop", linkageName: "loop", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 2, column: 3, scope: !5)
!9 = !DILocation(line: 3, column: 7, scope: !5)
!10 = !DILocation(line: 4, column: 3, scope: !5)
!11 = !DILocation(line: 5, column: 5, scope: !5)
!12 = !DILocation(line: 6, column: 3, scope: !5)
!13 = !DILocation(line: 99, column: 50, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
