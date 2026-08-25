; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DebugLine/DebugNoLine must not appear before OpPhi.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-line-if-phi.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V9:%[0-9]+]] = OpConstant [[I32]] 9{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}

; entry
; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V10]] [[V11]]
; CHECK-NEXT: OpSLessThan
; CHECK-NEXT: OpBranchConditional

; then
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V5]] [[V6]]
; CHECK-NEXT: OpIAdd
; CHECK-NEXT: OpBranch

; CHECK:      OpLabel
; CHECK-NEXT: [[PHI:%[0-9]+]] = OpPhi [[I32]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V9]] [[V9]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[PHI]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @if_fallthrough(i32 %x) !dbg !5 {
entry:
  %cmp = icmp slt i32 %x, 0, !dbg !8
  br i1 %cmp, label %then, label %merge, !dbg !8

then:
  %t = add i32 %x, 1, !dbg !9
  br label %merge, !dbg !9

merge:
  %r = phi i32 [ %t, %then ], [ %x, %entry ], !dbg !10
  ret i32 %r, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-line-if-phi.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "if_fallthrough", linkageName: "if_fallthrough", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 10, scope: !5)
!9 = !DILocation(line: 4, column: 5, scope: !5)
!10 = !DILocation(line: 8, column: 10, scope: !5)
!11 = !DILocation(line: 9, column: 3, scope: !5)
