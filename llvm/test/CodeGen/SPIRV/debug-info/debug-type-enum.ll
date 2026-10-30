; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A shader declares these enums and a local variable `Color c` in test():
;   enum Color { Red, Green = 5, Blue = -1 };
;   enum class Mode : uint { Off, On };
;   struct S { enum Kind { K = 2 }; };
; Each enum must get a DebugTypeEnum with its name, underlying type, line,
; Parent, size, flags and every enumerator. Blue's -1 prints as 4294967295
; because enumerator values are constants of an unsigned 32-bit type, and Mode
; carries FlagIsEnumClass (16384). The module also has an enum Bare whose
; metadata has no underlying type, which must use DebugInfoNone. The
; DebugLocalVariable for c must name Color as its type.

; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[str_uint:%[0-9]+]] = OpString "unsigned int"
; CHECK-SPIRV-DAG: [[str_Color:%[0-9]+]] = OpString "Color"
; CHECK-SPIRV-DAG: [[str_Red:%[0-9]+]] = OpString "Red"
; CHECK-SPIRV-DAG: [[str_Green:%[0-9]+]] = OpString "Green"
; CHECK-SPIRV-DAG: [[str_Blue:%[0-9]+]] = OpString "Blue"
; CHECK-SPIRV-DAG: [[str_Mode:%[0-9]+]] = OpString "Mode"
; CHECK-SPIRV-DAG: [[str_Off:%[0-9]+]] = OpString "Off"
; CHECK-SPIRV-DAG: [[str_On:%[0-9]+]] = OpString "On"
; CHECK-SPIRV-DAG: [[str_Bare:%[0-9]+]] = OpString "Bare"
; CHECK-SPIRV-DAG: [[str_Only:%[0-9]+]] = OpString "Only"
; CHECK-SPIRV-DAG: [[str_S:%[0-9]+]] = OpString "S"
; CHECK-SPIRV-DAG: [[str_Kind:%[0-9]+]] = OpString "Kind"
; CHECK-SPIRV-DAG: [[str_K:%[0-9]+]] = OpString "K"
; CHECK-SPIRV-DAG: [[str_c:%[0-9]+]] = OpString "c"
; CHECK-SPIRV-DAG: [[c0:%[0-9]+]] = OpConstant [[i32]] 0{{$}}
; CHECK-SPIRV-DAG: [[c1:%[0-9]+]] = OpConstant [[i32]] 1{{$}}
; CHECK-SPIRV-DAG: [[c2:%[0-9]+]] = OpConstant [[i32]] 2{{$}}
; CHECK-SPIRV-DAG: [[c3:%[0-9]+]] = OpConstant [[i32]] 3{{$}}
; CHECK-SPIRV-DAG: [[c4:%[0-9]+]] = OpConstant [[i32]] 4{{$}}
; CHECK-SPIRV-DAG: [[c5:%[0-9]+]] = OpConstant [[i32]] 5{{$}}
; CHECK-SPIRV-DAG: [[c6:%[0-9]+]] = OpConstant [[i32]] 6{{$}}
; CHECK-SPIRV-DAG: [[c7:%[0-9]+]] = OpConstant [[i32]] 7{{$}}
; CHECK-SPIRV-DAG: [[c8:%[0-9]+]] = OpConstant [[i32]] 8{{$}}
; CHECK-SPIRV-DAG: [[c11:%[0-9]+]] = OpConstant [[i32]] 11{{$}}
; CHECK-SPIRV-DAG: [[c32:%[0-9]+]] = OpConstant [[i32]] 32{{$}}
; CHECK-SPIRV-DAG: [[c16384:%[0-9]+]] = OpConstant [[i32]] 16384{{$}}
; CHECK-SPIRV-DAG: [[cneg1:%[0-9]+]] = OpConstant [[i32]] 4294967295{{$}}
; CHECK-SPIRV-DAG: [[dbgnone:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugInfoNone
; CHECK-SPIRV-DAG: [[ds:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource
; CHECK-SPIRV-DAG: [[cu:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugCompilationUnit
; CHECK-SPIRV-DAG: [[basic_int:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_int]]
; CHECK-SPIRV-DAG: [[basic_uint:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_uint]]
; CHECK-SPIRV-DAG: [[enum_Color:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Color]] [[basic_int]] [[ds]] [[c2]] [[c0]] [[cu]] [[c32]] [[c0]] [[c0]] [[str_Red]] [[c5]] [[str_Green]] [[cneg1]] [[str_Blue]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Mode]] [[basic_uint]] [[ds]] [[c3]] [[c0]] [[cu]] [[c32]] [[c16384]] [[c0]] [[str_Off]] [[c1]] [[str_On]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Bare]] [[dbgnone]] [[ds]] [[c4]] [[c0]] [[cu]] [[c32]] [[c0]] [[c7]] [[str_Only]]{{$}}

; An enum declared in a struct names the struct's DebugTypeComposite as Parent.
; CHECK-SPIRV-DAG: [[comp_S:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeComposite [[str_S]] [[c1]] [[ds]] [[c5]] [[c0]] [[cu]] {{%[0-9]+}} [[c8]] [[c0]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Kind]] [[basic_int]] [[ds]] [[c6]] [[c0]] [[comp_S]] [[c32]] [[c0]] [[c2]] [[str_K]]{{$}}

; CHECK-SPIRV-DAG: [[df:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugFunction
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugLocalVariable [[str_c]] [[enum_Color]] [[ds]] [[c11]] [[c0]] [[df]] [[c0]]{{$}}

define spir_func void @test() !dbg !21 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "enum.hlsl", directory: "/src")
!4 = !{!5, !10, !14, !17}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Color", file: !3, line: 2, baseType: !6, size: 32, elements: !7)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8, !9, !26}
!8 = !DIEnumerator(name: "Red", value: 0)
!9 = !DIEnumerator(name: "Green", value: 5)
!10 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Mode", file: !3, line: 3, baseType: !11, size: 32, flags: DIFlagEnumClass, elements: !12)
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!12 = !{!13, !27}
!13 = !DIEnumerator(name: "Off", value: 0, isUnsigned: true)
!14 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Bare", file: !3, line: 4, size: 32, elements: !15)
!15 = !{!16}
!16 = !DIEnumerator(name: "Only", value: 7)
!17 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Kind", scope: !18, file: !3, line: 6, baseType: !6, size: 32, elements: !19)
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 5, size: 8, elements: !20)
!19 = !{!28}
!20 = !{}
!21 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 10, type: !22, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !{!25}
!25 = !DILocalVariable(name: "c", scope: !21, file: !3, line: 11, type: !5)
!26 = !DIEnumerator(name: "Blue", value: -1)
!27 = !DIEnumerator(name: "On", value: 1, isUnsigned: true)
!28 = !DIEnumerator(name: "K", value: 2)
