; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise NonSemantic DebugLine emission:
;
;   OpExtInst %void %ext DebugLine %src %lineStart %lineEnd %colStart %colEnd
;
; with LineStart == LineEnd == DILocation::getLine(), ColStart == getCol(),
; ColEnd == getCol() + 1. All four are 32-bit integer OpConstants.
;
; DebugLine is an *in-block* instruction: it is emitted inline, in program order,
; immediately before the semantic instruction whose source location it describes,
; and applies to all following instructions until the next DebugLine, the next
; DebugNoLine, or the end of the block.
;

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-line.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}
; CHECK-DAG: [[V12:%[0-9]+]] = OpConstant [[I32]] 12{{$}}
; CHECK-DAG: [[V13:%[0-9]+]] = OpConstant [[I32]] 13{{$}}

; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: [[A:%[0-9]+]] = OpFunctionParameter
; CHECK-NEXT: [[B:%[0-9]+]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V10]] [[V11]]
; CHECK-NEXT: [[T0:%[0-9]+]] = OpIAdd [[I32]] [[A]] [[B]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V12]] [[V13]]
; CHECK-NEXT: [[T1:%[0-9]+]] = OpIMul [[I32]] [[T0]] [[A]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V5]] [[V5]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[T1]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_line(i32 %a, i32 %b) !dbg !5 {
entry:
  %t0 = add i32 %a, %b, !dbg !8   ; line 3, col 10
  %t1 = mul i32 %t0, %a, !dbg !9  ; line 4, col 12
  ret i32 %t1, !dbg !10           ; line 5, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-line.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "add_line", linkageName: "add_line", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 10, scope: !5)
!9 = !DILocation(line: 4, column: 12, scope: !5)
!10 = !DILocation(line: 5, column: 3, scope: !5)
