; asm-verbose=0 keeps AsmPrinter's ;DEBUG_VALUE: comments out of the output, so
; the CHECK-NEXT chain around the merge asserts adjacency of SPIR-V
; instructions rather than of text this backend does not own.
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DebugLine for the branch is emitted before OpSelectionMerge.
; DebugLine/DebugNoLine must not appear after the merge.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-line-selection-merge.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[CONDVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V7:%[0-9]+]] = OpConstant [[I32]] 7{{$}}
; CHECK-DAG: [[V20:%[0-9]+]] = OpConstant [[I32]] 20{{$}}
; CHECK-DAG: [[V21:%[0-9]+]] = OpConstant [[I32]] 21{{$}}
; CHECK-DAG: [[V9:%[0-9]+]] = OpConstant [[I32]] 9{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}
; CHECK-DAG: [[V50:%[0-9]+]] = OpConstant [[I32]] 50{{$}}
; CHECK-DAG: [[V51:%[0-9]+]] = OpConstant [[I32]] 51{{$}}
; CHECK-DAG: [[V99:%[0-9]+]] = OpConstant [[I32]] 99{{$}}

; entry
; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V10]] [[V11]]
; CHECK-NEXT: [[CMP:%[0-9]+]] = OpSLessThan
; A DebugValue between the merge and branch in MIR is emitted before the merge.
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugValue [[CONDVAR]] [[CMP]] [[EXPR]]{{ *$}}
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V99]] [[V99]] [[V50]] [[V51]]
; CHECK-NEXT: OpSelectionMerge
; CHECK-NEXT: OpBranchConditional

; else
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V6]] [[V6]] [[V5]] [[V6]]
; CHECK-NEXT: OpISub
; CHECK-NEXT: OpBranch

; then
; CHECK:      OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V5]] [[V6]]
; CHECK-NEXT: OpIAdd
; CHECK-NEXT: [[THENCOND:%[0-9]+]] = OpSLessThan
; Rebinding the variable in one successor must preserve the entry binding on
; the other successor.
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V7]] [[V7]] [[V20]] [[V21]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugValue [[CONDVAR]] [[THENCOND]] [[EXPR]]{{ *$}}
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V5]] [[V6]]
; CHECK-NEXT: OpSelect
; CHECK-NEXT: OpBranch

; merge
; CHECK:      OpLabel
; CHECK-NEXT: [[PHI:%[0-9]+]] = OpPhi [[I32]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V9]] [[V9]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[PHI]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @if_else(i32 %x) !dbg !5 {
entry:
  %cmp = icmp slt i32 %x, 0, !dbg !8
  call void @llvm.spv.selection.merge.p0(ptr blockaddress(@if_else, %merge), i32 0), !dbg !14
    #dbg_value(i1 %cmp, !16, !DIExpression(), !14)
  br i1 %cmp, label %then, label %else, !dbg !13

then:
  %sum = add i32 %x, 1, !dbg !9
  %thencond = icmp slt i32 %sum, 0, !dbg !9
    #dbg_value(i1 %thencond, !16, !DIExpression(), !17)
  ; %thencond needs a non-debug use, or the selector drops it as dead and the
  ; rebind this test is about never reaches the output.
  %t = select i1 %thencond, i32 %sum, i32 %x, !dbg !9
  br label %merge, !dbg !9

else:
  %e = sub i32 %x, 1, !dbg !10
  br label %merge, !dbg !10

merge:
  %r = phi i32 [ %t, %then ], [ %e, %else ], !dbg !11
  ret i32 %r, !dbg !12
}

declare void @llvm.spv.selection.merge.p0(ptr, i32 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-line-selection-merge.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "if_else", linkageName: "if_else", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 10, scope: !5)
!9 = !DILocation(line: 4, column: 5, scope: !5)
!10 = !DILocation(line: 6, column: 5, scope: !5)
!11 = !DILocation(line: 8, column: 10, scope: !5)
!12 = !DILocation(line: 9, column: 3, scope: !5)
!13 = !DILocation(line: 99, column: 50, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DIBasicType(name: "bool", size: 1, encoding: DW_ATE_boolean)
!16 = !DILocalVariable(name: "condition", scope: !5, file: !1, line: 7, type: !15)
; Column 20 keeps this record's DebugLine operands clear of the OpConstant 1
; the IR's own "add i32 %x, 1" already needs.
!17 = !DILocation(line: 7, column: 20, scope: !5)
