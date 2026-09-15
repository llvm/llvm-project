; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A struct, class, or union lowers to DebugTypeComposite, with one
; DebugTypeMember per data member. Members carry no Parent operand and are
; emitted before the composite that lists them. A forward-declared composite
; emits DebugInfoNone for Size and no members. Tag is 1 for a structure.

; CHECK-SPIRV: OpCapability Int64
; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[i64:%[0-9]+]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[str_float:%[0-9]+]] = OpString "float"
; CHECK-SPIRV-DAG: [[str_S:%[0-9]+]] = OpString "S"
; CHECK-SPIRV-DAG: [[str_a:%[0-9]+]] = OpString "a"
; CHECK-SPIRV-DAG: [[str_b:%[0-9]+]] = OpString "b"
; CHECK-SPIRV-DAG: [[str_Fwd:%[0-9]+]] = OpString "Fwd"
; CHECK-SPIRV-DAG: [[str_Wide:%[0-9]+]] = OpString "Wide"
; CHECK-SPIRV-DAG: [[str_wide:%[0-9]+]] = OpString "wide"
; CHECK-SPIRV-DAG: [[str_pad:%[0-9]+]] = OpString "pad"
; CHECK-SPIRV-DAG: [[str_tail:%[0-9]+]] = OpString "tail"
; CHECK-SPIRV-DAG: [[str_u32max:%[0-9]+]] = OpString "u32max"
; CHECK-SPIRV-DAG: [[str_edge:%[0-9]+]] = OpString "edge"
; CHECK-SPIRV-DAG: [[c0:%[0-9]+]] = OpConstant [[i32]] 0{{$}}
; CHECK-SPIRV-DAG: [[c1:%[0-9]+]] = OpConstant [[i32]] 1{{$}}
; CHECK-SPIRV-DAG: [[c32:%[0-9]+]] = OpConstant [[i32]] 32{{$}}
; CHECK-SPIRV-DAG: [[c64:%[0-9]+]] = OpConstant [[i32]] 64{{$}}
; CHECK-SPIRV-DAG: [[c16:%[0-9]+]] = OpConstant [[i32]] 16{{$}}
; CHECK-SPIRV-DAG: [[dbgnone:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugInfoNone
; CHECK-SPIRV-DAG: [[ds:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource
; CHECK-SPIRV-DAG: [[cu:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugCompilationUnit
; CHECK-SPIRV-DAG: [[basic_int:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_int]]
; CHECK-SPIRV-DAG: [[basic_float:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_float]]
; CHECK-SPIRV-DAG: [[member_a:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_a]] [[basic_int]] [[ds]] {{%[0-9]+}} [[c0]] [[c0]] [[c32]] [[c0]]{{$}}
; CHECK-SPIRV-DAG: [[member_b:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_b]] [[basic_float]] [[ds]] {{%[0-9]+}} [[c0]] [[c32]] [[c32]] [[c0]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeComposite [[str_S]] [[c1]] [[ds]] {{%[0-9]+}} [[c0]] [[cu]] {{%[0-9]+}} [[c64]] [[c0]] [[member_a]] [[member_b]]{{$}}
; A forward declaration carries FlagFwdDecl (16) in its Flags operand.
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeComposite [[str_Fwd]] [[c1]] [[ds]] {{%[0-9]+}} [[c0]] [[cu]] {{%[0-9]+}} [[dbgnone]] [[c16]]{{$}}

; CHECK-SPIRV-DAG: [[big:%[0-9]+]] = OpConstant [[i64]] 8589934592{{$}}
; CHECK-SPIRV-DAG: [[bigsum:%[0-9]+]] = OpConstant [[i64]] 8589934624{{$}}
; CHECK-SPIRV-DAG: [[basic_wide:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_wide]] [[big]] {{%[0-9]+}} [[c0]]{{$}}
; CHECK-SPIRV-DAG: [[member_pad:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_pad]] [[basic_wide]] [[ds]] {{%[0-9]+}} [[c0]] [[c0]] [[big]] [[c0]]{{$}}
; CHECK-SPIRV-DAG: [[member_tail:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_tail]] [[basic_int]] [[ds]] {{%[0-9]+}} [[c0]] [[big]] [[c32]] [[c0]]{{$}}

; CHECK-SPIRV-DAG: [[cu32max:%[0-9]+]] = OpConstant [[i32]] 4294967295{{$}}
; CHECK-SPIRV-DAG: [[bigtotal:%[0-9]+]] = OpConstant [[i64]] 12884901919{{$}}
; CHECK-SPIRV-DAG: [[basic_u32max:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_u32max]] [[cu32max]] {{%[0-9]+}} [[c0]]{{$}}
; CHECK-SPIRV-DAG: [[member_edge:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_edge]] [[basic_u32max]] [[ds]] {{%[0-9]+}} [[c0]] [[bigsum]] [[cu32max]] [[c0]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeComposite [[str_Wide]] [[c1]] [[ds]] {{%[0-9]+}} [[c0]] [[cu]] {{%[0-9]+}} [[bigtotal]] [[c0]] [[member_pad]] [[member_tail]] [[member_edge]]{{$}}

define spir_func void @test() !dbg !13 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "composite.hlsl", directory: "/src")
!4 = !{!5, !12, !16}
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 64, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", file: !3, line: 2, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", file: !3, line: 3, baseType: !10, size: 32, offset: 32)
!10 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "Fwd", file: !3, line: 5, flags: DIFlagFwdDecl)
!13 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 10, type: !14, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "Wide", file: !3, line: 7, size: 12884901919, elements: !17)
!17 = !{!18, !20, !21}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "pad", file: !3, line: 8, baseType: !19, size: 8589934592)
!19 = !DIBasicType(name: "wide", size: 8589934592, encoding: DW_ATE_unsigned)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "tail", file: !3, line: 9, baseType: !8, size: 32, offset: 8589934592)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "edge", file: !3, line: 10, baseType: !22, size: 4294967295, offset: 8589934624)
!22 = !DIBasicType(name: "u32max", size: 4294967295, encoding: DW_ATE_unsigned)
