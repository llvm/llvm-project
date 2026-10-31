; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Two shaders, first.hlsl and second.hlsl, are linked into one module, and each
; has its own DebugCompilationUnit. Both include enums.h: first.hlsl uses enum
; First and second.hlsl uses enum Second. Each DebugTypeEnum must name its own
; shader's DebugCompilationUnit as Parent. Both shaders also use enum Shared,
; which linking merges into one node that both units list. Its one
; DebugTypeEnum must name first.hlsl's unit, the first unit that lists it.

; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[str_first_file:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}first.hlsl"
; CHECK-SPIRV-DAG: [[str_second_file:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}second.hlsl"
; CHECK-SPIRV-DAG: [[str_header:%[0-9]+]] = OpString "{{[/\\]}}include{{[/\\]}}enums.h"
; CHECK-SPIRV-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-SPIRV-DAG: [[str_First:%[0-9]+]] = OpString "First"
; CHECK-SPIRV-DAG: [[str_Second:%[0-9]+]] = OpString "Second"
; CHECK-SPIRV-DAG: [[str_Shared:%[0-9]+]] = OpString "Shared"
; CHECK-SPIRV-DAG: [[str_Zero:%[0-9]+]] = OpString "Zero"
; CHECK-SPIRV-DAG: [[c0:%[0-9]+]] = OpConstant [[i32]] 0{{$}}
; CHECK-SPIRV-DAG: [[c1:%[0-9]+]] = OpConstant [[i32]] 1{{$}}
; CHECK-SPIRV-DAG: [[c2:%[0-9]+]] = OpConstant [[i32]] 2{{$}}
; CHECK-SPIRV-DAG: [[c32:%[0-9]+]] = OpConstant [[i32]] 32{{$}}
; CHECK-SPIRV-DAG: [[source_first:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource [[str_first_file]]{{$}}
; CHECK-SPIRV-DAG: [[source_second:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource [[str_second_file]]{{$}}
; CHECK-SPIRV-DAG: [[source_header:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugSource [[str_header]]{{$}}
; CHECK-SPIRV-DAG: [[cu_first:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugCompilationUnit {{%[0-9]+}} {{%[0-9]+}} [[source_first]] {{%[0-9]+}}{{$}}
; CHECK-SPIRV-DAG: [[cu_second:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugCompilationUnit {{%[0-9]+}} {{%[0-9]+}} [[source_second]] {{%[0-9]+}}{{$}}
; CHECK-SPIRV-DAG: [[basic_int:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeBasic [[str_int]]
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_First]] [[basic_int]] [[source_header]] [[c1]] [[c0]] [[cu_first]] [[c32]] [[c0]] [[c0]] [[str_Zero]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Second]] [[basic_int]] [[source_header]] [[c1]] [[c0]] [[cu_second]] [[c32]] [[c0]] [[c0]] [[str_Zero]]{{$}}
; CHECK-SPIRV-DAG: OpExtInst [[void]] [[ext]] DebugTypeEnum [[str_Shared]] [[basic_int]] [[source_header]] [[c2]] [[c0]] [[cu_first]] [[c32]] [[c0]] [[c0]] [[str_Zero]]{{$}}

define spir_func void @first() !dbg !12 {
entry:
  ret void
}

define spir_func void @second() !dbg !13 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0, !1}
!llvm.module.flags = !{!14, !15}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5)
!1 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6)
!2 = !DIFile(filename: "first.hlsl", directory: "/src")
!3 = !DIFile(filename: "second.hlsl", directory: "/src")
!4 = !DIFile(filename: "enums.h", directory: "/include")
!5 = !{!7, !18}
!6 = !{!8, !18}
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "First", file: !4, line: 1, baseType: !9, size: 32, elements: !10)
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Second", file: !4, line: 1, baseType: !9, size: 32, elements: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DIEnumerator(name: "Zero", value: 0)
!12 = distinct !DISubprogram(name: "first", scope: !2, file: !2, line: 1, type: !16, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!13 = distinct !DISubprogram(name: "second", scope: !3, file: !3, line: 1, type: !16, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Shared", file: !4, line: 2, baseType: !9, size: 32, elements: !10)
