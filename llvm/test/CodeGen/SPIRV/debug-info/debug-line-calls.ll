; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-line-calls.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[NAME_INC:%[0-9]+]] = OpString "inc"
; CHECK-DAG: [[NAME_CALLER:%[0-9]+]] = OpString "caller"
; CHECK-DAG: [[DF_INC:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_INC]]
; CHECK-DAG: [[DF_CALLER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_CALLER]]
; CHECK-DAG: [[V2:%[0-9]+]] = OpConstant [[I32]] 2{{$}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V7:%[0-9]+]] = OpConstant [[I32]] 7{{$}}
; CHECK-DAG: [[V8:%[0-9]+]] = OpConstant [[I32]] 8{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}
; CHECK-DAG: [[V12:%[0-9]+]] = OpConstant [[I32]] 12{{$}}
; CHECK-DAG: [[V13:%[0-9]+]] = OpConstant [[I32]] 13{{$}}

; inc: add and return.
; CHECK:      [[INC:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_INC]] [[INC]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V2]] [[V2]] [[V10]] [[V11]]
; CHECK-NEXT: [[T0:%[0-9]+]] = OpIAdd [[I32]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[T0]]
; CHECK-NEXT: OpFunctionEnd

; caller: two calls to inc, then return.
; CHECK:      [[CALLER:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_CALLER]] [[CALLER]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V6]] [[V6]] [[V12]] [[V13]]
; CHECK-NEXT: OpFunctionCall [[I32]] [[INC]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V7]] [[V7]] [[V12]] [[V13]]
; CHECK-NEXT: OpFunctionCall [[I32]] [[INC]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V8]] [[V8]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @inc(i32 %x) !dbg !5 {
entry:
  %r = add i32 %x, 1, !dbg !8   ; line 2, col 10
  ret i32 %r, !dbg !9           ; line 3, col 3
}

define spir_func i32 @caller(i32 %x) !dbg !10 {
entry:
  %a = call spir_func i32 @inc(i32 %x), !dbg !13   ; line 6, col 12
  %b = call spir_func i32 @inc(i32 %a), !dbg !14   ; line 7, col 12
  ret i32 %b, !dbg !15                              ; line 8, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-line-calls.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "inc", linkageName: "inc", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 2, column: 10, scope: !5)
!9 = !DILocation(line: 3, column: 3, scope: !5)

!10 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 6, column: 12, scope: !10)
!14 = !DILocation(line: 7, column: 12, scope: !10)
!15 = !DILocation(line: 8, column: 3, scope: !10)
