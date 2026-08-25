; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise DebugLine re-emission at basic block boundaries.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-line-block.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}

; entry block: add and br share line 3.
; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V10]] [[V11]]
; CHECK-NEXT: [[T0:%[0-9]+]] = OpIAdd [[I32]]
; CHECK-NOT:  OpExtInst [[VOID]] [[EXT]] DebugLine
; CHECK-NEXT: OpBranch

; then block: mul reuses line 3; region must be reopened at the block boundary.
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V10]] [[V11]]
; CHECK-NEXT: [[T1:%[0-9]+]] = OpIMul [[I32]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V5]] [[V5]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[T1]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @block_boundary(i32 %x) !dbg !5 {
entry:
  %t0 = add i32 %x, 1, !dbg !8   ; line 3, col 10
  br label %then, !dbg !8        ; line 3, col 10 (same region as add)

then:
  %t1 = mul i32 %t0, %t0, !dbg !8 ; line 3, col 10 (same loc, new block)
  ret i32 %t1, !dbg !9             ; line 5, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-line-block.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "block_boundary", linkageName: "block_boundary", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 10, scope: !5)
!9 = !DILocation(line: 5, column: 3, scope: !5)
