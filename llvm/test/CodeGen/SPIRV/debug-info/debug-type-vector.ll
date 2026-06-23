; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that DICompositeType nodes with tag DW_TAG_array_type and the
; DIFlagVector flag lower to DebugTypeVector. Three element widths exercise
; the OpConstant cache: a 4-component float vector and a 3-component float
; vector share the float DebugTypeBasic but have distinct component-count
; constants, while a 4-component int vector reuses the OpConstant 4 already
; emitted for the float4.
;
; The OpConstant cache merges values across uses. OpConstant 3 is emitted
; once and shared between the float DebugTypeBasic's Encoding operand (3 =
; Float, NSDI 4.5) and the 3-component vector's ComponentCount operand.
; Likewise OpConstant 4 is shared between the int DebugTypeBasic's Encoding
; (4 = Signed) and the 4-component vector's ComponentCount.

; CHECK-SPIRV: [[ext_inst_non_semantic:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[str_float:%[0-9]+]] = OpString "float"
; CHECK-SPIRV-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[size_32bit:%[0-9]+]] = OpConstant [[type_int32]] 32{{$}}
; CHECK-SPIRV-DAG: [[flag_zero:%[0-9]+]] = OpConstant [[type_int32]] 0{{$}}
; CHECK-SPIRV-DAG: [[const_3:%[0-9]+]] = OpConstant [[type_int32]] 3{{$}}
; CHECK-SPIRV-DAG: [[const_4:%[0-9]+]] = OpConstant [[type_int32]] 4{{$}}
; CHECK-SPIRV-DAG: [[basic_float:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_float]] [[size_32bit]] [[const_3]] [[flag_zero]]
; CHECK-SPIRV-DAG: [[basic_int:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeBasic [[str_int]] [[size_32bit]] [[const_4]] [[flag_zero]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeVector [[basic_float]] [[const_4]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeVector [[basic_float]] [[const_3]]
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeVector [[basic_int]] [[const_4]]

define spir_func void @test() !dbg !6 {
entry:
  %f4 = alloca <4 x float>, align 16
  %f3 = alloca <3 x float>, align 16
  %i4 = alloca <4 x i32>, align 16
    #dbg_declare(ptr %f4, !10, !DIExpression(), !14)
    #dbg_declare(ptr %f3, !15, !DIExpression(), !17)
    #dbg_declare(ptr %i4, !18, !DIExpression(), !22)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "vector.hlsl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocalVariable(name: "f4", scope: !6, file: !1, line: 2, type: !11)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 128, flags: DIFlagVector, elements: !13)
!12 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !{!DISubrange(count: 4)}
!14 = !DILocation(line: 2, column: 10, scope: !6)
!15 = !DILocalVariable(name: "f3", scope: !6, file: !1, line: 3, type: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 96, flags: DIFlagVector, elements: !20)
!17 = !DILocation(line: 3, column: 10, scope: !6)
!18 = !DILocalVariable(name: "i4", scope: !6, file: !1, line: 4, type: !19)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 128, flags: DIFlagVector, elements: !13)
!20 = !{!DISubrange(count: 3)}
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !DILocation(line: 4, column: 8, scope: !6)
