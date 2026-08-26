; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypeMatrix
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A DW_TAG_array_type without DIFlagVector lowers to DebugTypeArray, one
; component count per subrange. A runtime-sized subrange emits count 0. An
; array of a vector references the element's DebugTypeVector.
;
; Clang emits matrix types two-subrange DW_TAG_array_type, so it lowers to
; DebugTypeArray. This is checked by --implicit-check-not=DebugTypeMatrix.
; If Clang starts emitting matrix types, this will fail.

; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[str_float:%[0-9]+]] = OpString "float"
; CHECK-SPIRV-DAG: [[c0:%[0-9]+]] = OpConstant [[i32]] 0{{$}}
; CHECK-SPIRV-DAG: [[c3:%[0-9]+]] = OpConstant [[i32]] 3{{$}}
; CHECK-SPIRV-DAG: [[c4:%[0-9]+]] = OpConstant [[i32]] 4{{$}}
; CHECK-SPIRV-DAG: [[basic_float:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_float]]
; CHECK-SPIRV-DAG: [[vec_float3:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeVector [[basic_float]] [[c3]]
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeArray [[basic_float]] [[c4]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeArray [[basic_float]] [[c4]] [[c3]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeArray [[basic_float]] [[c0]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeArray [[vec_float3]] [[c4]]{{$}}

define spir_func void @test() !dbg !6 {
entry:
  %a1 = alloca [4 x float], align 16
  %a2 = alloca [3 x [4 x float]], align 16
  %ar = alloca [0 x float], align 16
  %av = alloca [4 x <3 x float>], align 16
    #dbg_declare(ptr %a1, !10, !DIExpression(), !14)
    #dbg_declare(ptr %a2, !15, !DIExpression(), !17)
    #dbg_declare(ptr %ar, !18, !DIExpression(), !19)
    #dbg_declare(ptr %av, !23, !DIExpression(), !26)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "array.hlsl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocalVariable(name: "a1", scope: !6, file: !1, line: 2, type: !11)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 128, elements: !13)
!12 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !{!DISubrange(count: 4)}
!14 = !DILocation(line: 2, column: 10, scope: !6)
!15 = !DILocalVariable(name: "a2", scope: !6, file: !1, line: 3, type: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 384, elements: !20)
!17 = !DILocation(line: 3, column: 10, scope: !6)
!18 = !DILocalVariable(name: "ar", scope: !6, file: !1, line: 4, type: !21)
!19 = !DILocation(line: 4, column: 8, scope: !6)
!20 = !{!DISubrange(count: 4), !DISubrange(count: 3)}
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 0, elements: !22)
!22 = !{!DISubrange(lowerBound: 0)}
!23 = !DILocalVariable(name: "av", scope: !6, file: !1, line: 5, type: !24)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !25, size: 384, elements: !27)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 96, flags: DIFlagVector, elements: !28)
!26 = !DILocation(line: 5, column: 8, scope: !6)
!27 = !{!DISubrange(count: 4)}
!28 = !{!DISubrange(count: 3)}
