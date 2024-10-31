; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; 
; TODO: spirv-val emits "line 42: NonSemantic.Shader.DebugInfo.100 DebugCompilationUnit:
; expected operand Language must be a result id of 32-bit unsigned OpConstant" that's because
; OpConstant with zero is substituted by OpConstantNull in resulting SPIR-V output. It's unrelated
; to debug-info.  
; DISABLED: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR-DAG: [[void:%[0-9]+\:[a-z\(\)0-9]+]] = OpTypeVoid
; CHECK-MIR-DAG: [[i32:%[0-9]+\:[a-z\(\)0-9]+]] = OpTypeInt 32, 0
; CHECK-MIR-DAG: [[i32_1:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 1
; CHECK-MIR-DAG: [[i32_6:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 6
; CHECK-MIR-DAG: [[i32_10:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 10
; CHECK-MIR-DAG: [[i32_5:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 5
; CHECK-MIR-DAG: [[i32_3:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 3
; CHECK-MIR-DAG: [[i32_32:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 32
; CHECK-MIR-DAG: [[i32_4:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 4
; CHECK-MIR-DAG: [[i32_11:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 11
; CHECK-MIR-DAG: [[i32_2:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 2
; CHECK-MIR-DAG: [[i32_7:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 7
; CHECK-MIR-DAG: [[i32_14:%[0-9]+\:[a-z\(\)0-9]+]] = OpConstantI [[i32]], 14
; CHECK-MIR-DAG: [[file_scope:%[0-9]+\:[a-z\(\)0-9]+]] = OpExtInst [[void]], 3, 35
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_1]], [[i32_1]], [[i32_14]], [[i32_14]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_2]], [[i32_2]], [[i32_7]], [[i32_7]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_2]], [[i32_2]], [[i32_11]], [[i32_11]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_3]], [[i32_3]], [[i32_7]], [[i32_7]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_4]], [[i32_4]], [[i32_7]], [[i32_7]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_5]], [[i32_5]], [[i32_7]], [[i32_7]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_6]], [[i32_6]], [[i32_10]], [[i32_10]]
; CHECK-MIR-DAG: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_6]], [[i32_6]], [[i32_3]], [[i32_3]]
;
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_1]], [[i32_1]], [[i32_14]], [[i32_14]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_2]], [[i32_2]], [[i32_7]], [[i32_7]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_2]], [[i32_2]], [[i32_11]], [[i32_11]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_3]], [[i32_3]], [[i32_7]], [[i32_7]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_4]], [[i32_4]], [[i32_7]], [[i32_7]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_5]], [[i32_5]], [[i32_7]], [[i32_7]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_6]], [[i32_6]], [[i32_10]], [[i32_10]]
; CHECK-MIR-NOT: OpExtInst [[void]], 3, 103, [[file_scope]], [[i32_6]], [[i32_6]], [[i32_3]], [[i32_3]]

; CHECK-SPIRV-DAG: [[nonsemantic_di:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32_1:%[0-9]+]] = OpConstant [[i32]] 1
; CHECK-SPIRV-DAG: [[i32_5:%[0-9]+]] = OpConstant [[i32]] 5
; CHECK-SPIRV-DAG: [[i32_3:%[0-9]+]] = OpConstant [[i32]] 3
; CHECK-SPIRV-DAG: [[i32_32:%[0-9]+]] = OpConstant [[i32]] 32
; CHECK-SPIRV-DAG: [[i32_4:%[0-9]+]] = OpConstant [[i32]] 4
; CHECK-SPIRV-DAG: [[i32_14:%[0-9]+]] = OpConstant [[i32]] 14
; CHECK-SPIRV-DAG: [[i32_2:%[0-9]+]] = OpConstant [[i32]] 2
; CHECK-SPIRV-DAG: [[i32_7:%[0-9]+]] = OpConstant [[i32]] 7
; CHECK-SPIRV-DAG: [[i32_11:%[0-9]+]] = OpConstant [[i32]] 11
; CHECK-SPIRV-DAG: [[i32_6:%[0-9]+]] = OpConstant [[i32]] 6
; CHECK-SPIRV-DAG: [[i32_10:%[0-9]+]] = OpConstant [[i32]] 10
; CHECK-SPIRV: [[file_source:%[0-9]+]] = OpExtInst [[void]] [[nonsemantic_di]] DebugSource
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_1]] [[i32_1]] [[i32_14]] [[i32_14]]
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_2]] [[i32_2]] [[i32_7]] [[i32_7]]
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_2]] [[i32_2]] [[i32_11]] [[i32_11]]
; CHECK-SPIRV: OpLoad
; CHECK-SPIRV: OpIAdd
; CHECK-SPIRV: OpStore
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_3]] [[i32_3]] [[i32_7]] [[i32_7]]
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_3]] [[i32_3]] [[i32_11]] [[i32_11]]
; CHECK-SPIRV: OpLoad
; CHECK-SPIRV: OpIAdd
; CHECK-SPIRV: OpStore
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_4]] [[i32_4]] [[i32_7]] [[i32_7]]
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_4]] [[i32_4]] [[i32_11]] [[i32_11]]
; CHECK-SPIRV: OpLoad
; CHECK-SPIRV: OpIAdd
; CHECK-SPIRV: OpStore
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_5]] [[i32_5]] [[i32_7]] [[i32_7]]
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_5]] [[i32_5]] [[i32_11]] [[i32_11]]
; CHECK-SPIRV: OpLoad
; CHECK-SPIRV: OpIAdd
; CHECK-SPIRV: OpStore
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_6]] [[i32_6]] [[i32_10]] [[i32_10]]
; CHECK-SPIRV: OpLoad
; CHECK-SPIRV: OpExtInst [[void]] [[nonsemantic_di]] DebugLine [[file_source]] [[i32_6]] [[i32_6]] [[i32_3]] [[i32_3]]

define dso_local spir_func i32 @test(i32 noundef %a) #0 !dbg !6 {
entry:
  %a.addr = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
    #dbg_declare(ptr %a.addr, !12, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !13)
    #dbg_declare(ptr %b, !14, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !15)
  %0 = load i32, ptr %a.addr, align 4, !dbg !16
  %add = add nsw i32 %0, 1, !dbg !17
  store i32 %add, ptr %b, align 4, !dbg !15
    #dbg_declare(ptr %c, !18, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !19)
  %1 = load i32, ptr %b, align 4, !dbg !20
  %add1 = add nsw i32 %1, 1, !dbg !21
  store i32 %add1, ptr %c, align 4, !dbg !19
    #dbg_declare(ptr %d, !22, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !23)
  %2 = load i32, ptr %c, align 4, !dbg !24
  %add2 = add nsw i32 %2, 1, !dbg !25
  store i32 %add2, ptr %d, align 4, !dbg !23
    #dbg_declare(ptr %e, !26, !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), !27)
  %3 = load i32, ptr %d, align 4, !dbg !28
  %add3 = add nsw i32 %3, 1, !dbg !29
  store i32 %add3, ptr %e, align 4, !dbg !27
  %4 = load i32, ptr %e, align 4, !dbg !30
  ret i32 %4, !dbg !31
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version XX.X.Xgit (GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG 9999999999999999999999999999999999999999)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "YYYYYYY", directory: "/XXXXXXXXXXXXXXXX/XXXXXXXXXXXXXXXX/XXXXXXXXXXX", checksumkind: CSK_MD5, checksum: "66666666666666666666666666666666")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "test", scope: !7, file: !7, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!7 = !DIFile(filename: "ZZZZZ", directory: "YYYYYYYYY/YYYYYYYYYYYY/YYYYYYYYYYYYY/YYYYYYYYY", checksumkind: CSK_MD5, checksum: "77777777777777777777777777777777")
!8 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !6, file: !7, line: 1, type: !10)
!13 = !DILocation(line: 1, column: 14, scope: !6)
!14 = !DILocalVariable(name: "b", scope: !6, file: !7, line: 2, type: !10)
!15 = !DILocation(line: 2, column: 7, scope: !6)
!16 = !DILocation(line: 2, column: 11, scope: !6)
!17 = !DILocation(line: 2, column: 13, scope: !6)
!18 = !DILocalVariable(name: "c", scope: !6, file: !7, line: 3, type: !10)
!19 = !DILocation(line: 3, column: 7, scope: !6)
!20 = !DILocation(line: 3, column: 11, scope: !6)
!21 = !DILocation(line: 3, column: 13, scope: !6)
!22 = !DILocalVariable(name: "d", scope: !6, file: !7, line: 4, type: !10)
!23 = !DILocation(line: 4, column: 7, scope: !6)
!24 = !DILocation(line: 4, column: 11, scope: !6)
!25 = !DILocation(line: 4, column: 13, scope: !6)
!26 = !DILocalVariable(name: "e", scope: !6, file: !7, line: 5, type: !10)
!27 = !DILocation(line: 5, column: 7, scope: !6)
!28 = !DILocation(line: 5, column: 11, scope: !6)
!29 = !DILocation(line: 5, column: 13, scope: !6)
!30 = !DILocation(line: 6, column: 10, scope: !6)
!31 = !DILocation(line: 6, column: 3, scope: !6)
