; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypedef
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Typedefs are emitted in a single DebugInfoFinder pass. Bar is discovered
; before Foo, so when Bar is emitted its Foo base has no id yet and Bar is
; dropped. Foo is emitted next and survives. Only one DebugTypedef is emitted
; here, for Foo. Emitting typedefs in dependency order, would add a second
; DebugTypedef for Bar (see https://github.com/llvm/llvm-project/issues/211850).

; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[str_foo:%[0-9]+]] = OpString "Foo"
; CHECK-SPIRV-DAG: [[c0:%[0-9]+]] = OpConstant [[i32]] 0{{$}}
; CHECK-SPIRV-DAG: [[c2:%[0-9]+]] = OpConstant [[i32]] 2{{$}}
; CHECK-SPIRV-DAG: [[ds:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource
; CHECK-SPIRV-DAG: [[cu:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugCompilationUnit
; CHECK-SPIRV-DAG: [[basic_int:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_int]]
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypedef [[str_foo]] [[basic_int]] [[ds]] [[c2]] [[c0]] [[cu]]{{$}}

define spir_func void @test() !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "typedef.hlsl", directory: "/src")

; Only Bar is retained. Foo is reached solely through Bar's base type, so the
; finder visits Bar first and Foo second.
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar", file: !3, line: 3, baseType: !7, scope: !3)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "Foo", file: !3, line: 2, baseType: !9, scope: !3)
!8 = !DISubroutineType(types: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{null}
