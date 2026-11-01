; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt
; RUN: FileCheck %s --check-prefix=CHECK-SPIRV --input-file %t.spvt
; RUN: FileCheck %s --check-prefix=NO-FWD --input-file %t.spvt
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A shader declares two 64-bit enums, each with one enumerator that needs more
; than 32 bits:
;   enum BigS : int64_t  { Small = 1, Huge = 2147483648 };
;   enum BigU : uint64_t { Max = 4294967295, Over = 4294967296 };
; DebugTypeEnum stores enumerator values as 32-bit integers, so BigS must list
; only Small and BigU only Max. Huge would fit as unsigned, but BigS is signed.
; A local variable b of type BigS must still name BigS as its type.

; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[str_ll:%[0-9]+]] = OpString "long long"
; CHECK-SPIRV-DAG: [[str_ull:%[0-9]+]] = OpString "unsigned long long"
; CHECK-SPIRV-DAG: [[str_BigS:%[0-9]+]] = OpString "BigS"
; CHECK-SPIRV-DAG: [[str_Small:%[0-9]+]] = OpString "Small"
; CHECK-SPIRV-DAG: [[str_BigU:%[0-9]+]] = OpString "BigU"
; CHECK-SPIRV-DAG: [[str_Max:%[0-9]+]] = OpString "Max"
; CHECK-SPIRV-DAG: [[str_b:%[0-9]+]] = OpString "b"
; CHECK-SPIRV-DAG: [[c0:%[0-9]+]] = OpConstant [[i32]] 0{{$}}
; CHECK-SPIRV-DAG: [[c1:%[0-9]+]] = OpConstant [[i32]] 1{{$}}
; CHECK-SPIRV-DAG: [[c2:%[0-9]+]] = OpConstant [[i32]] 2{{$}}
; CHECK-SPIRV-DAG: [[c3:%[0-9]+]] = OpConstant [[i32]] 3{{$}}
; CHECK-SPIRV-DAG: [[c11:%[0-9]+]] = OpConstant [[i32]] 11{{$}}
; CHECK-SPIRV-DAG: [[c64:%[0-9]+]] = OpConstant [[i32]] 64{{$}}
; CHECK-SPIRV-DAG: [[cmax:%[0-9]+]] = OpConstant [[i32]] 4294967295{{$}}
; CHECK-SPIRV-DAG: [[ds:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource
; CHECK-SPIRV-DAG: [[cu:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugCompilationUnit
; CHECK-SPIRV-DAG: [[basic_ll:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_ll]]
; CHECK-SPIRV-DAG: [[basic_ull:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_ull]]
; CHECK-SPIRV-DAG: [[enum_BigS:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_BigS]] [[basic_ll]] [[ds]] [[c2]] [[c0]] [[cu]] [[c64]] [[c0]] [[c1]] [[str_Small]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_BigU]] [[basic_ull]] [[ds]] [[c3]] [[c0]] [[cu]] [[c64]] [[c0]] [[cmax]] [[str_Max]]{{$}}
; CHECK-SPIRV-DAG: [[df:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugFunction
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugLocalVariable [[str_b]] [[enum_BigS]] [[ds]] [[c11]] [[c0]] [[df]] [[c0]]{{$}}

; The metadata of enum Mixed lists a DISubrange beside its one enumerator,
; Only = 3. The verifier accepts any node in an enum's elements, and the
; DebugTypeEnum must list only the enumerator.
; CHECK-SPIRV-DAG: [[str_Mixed:%[0-9]+]] = OpString "Mixed"
; CHECK-SPIRV-DAG: [[str_Only:%[0-9]+]] = OpString "Only"
; CHECK-SPIRV-DAG: [[c5:%[0-9]+]] = OpConstant [[i32]] 5{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Mixed]] [[basic_ll]] [[ds]] [[c5]] [[c0]] [[cu]] [[c64]] [[c0]] [[c3]] [[str_Only]]{{$}}

; The module also has enum Fwd, declared but never defined, so its size is 0,
; and a local variable f of type Fwd. DebugTypeEnum needs a size, so neither Fwd
; nor f may appear in the debug info. The OpStrings precede every OpExtInst, so
; the NOT lines cover all debug instructions.
; NO-FWD-DAG: [[str_Fwd:%[0-9]+]] = OpString "Fwd"
; NO-FWD-DAG: [[str_f:%[0-9]+]] = OpString "f"
; NO-FWD-NOT: DebugTypeEnum [[str_Fwd]] %
; NO-FWD-NOT: DebugLocalVariable [[str_f]] %

define spir_func void @test() !dbg !17 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "enum.hlsl", directory: "/src")
!4 = !{!5, !10, !15, !23}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "BigS", file: !3, line: 2, baseType: !6, size: 64, elements: !7)
!6 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "Small", value: 1)
!9 = !DIEnumerator(name: "Huge", value: 2147483648)
!10 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "BigU", file: !3, line: 3, baseType: !11, size: 64, elements: !12)
!11 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!12 = !{!13, !14}
!13 = !DIEnumerator(name: "Max", value: 4294967295, isUnsigned: true)
!14 = !DIEnumerator(name: "Over", value: 4294967296, isUnsigned: true)
!15 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Fwd", file: !3, line: 4, flags: DIFlagFwdDecl, identifier: "Fwd")
!17 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 10, type: !18, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{!21, !22}
!21 = !DILocalVariable(name: "b", scope: !17, file: !3, line: 11, type: !5)
!22 = !DILocalVariable(name: "f", scope: !17, file: !3, line: 12, type: !15)
!23 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Mixed", file: !3, line: 5, baseType: !6, size: 64, elements: !24)
!24 = !{!25, !26}
!25 = !DISubrange(count: 4)
!26 = !DIEnumerator(name: "Only", value: 3)
