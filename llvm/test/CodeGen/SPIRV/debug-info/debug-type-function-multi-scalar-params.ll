; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
;
; Multi-slot signature mapping order: DebugTypeFunction must keep operand order
; for void(int, float, int).

; CHECK: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[type_int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[flag_zero:%[0-9]+]] = OpConstant [[type_int32]] 0{{$}}
; CHECK-DAG: [[str_int:%[0-9]+]] = OpString "int"
; CHECK-DAG: [[str_float:%[0-9]+]] = OpString "float"
; CHECK-DAG: [[dbg_int:%[0-9]+]] = OpExtInst [[type_void]] [[ext]] DebugTypeBasic [[str_int]]
; CHECK-DAG: [[dbg_float:%[0-9]+]] = OpExtInst [[type_void]] [[ext]] DebugTypeBasic [[str_float]]
; CHECK: OpExtInst [[type_void]] [[ext]] DebugTypeFunction [[flag_zero]] [[type_void]] [[dbg_int]] [[dbg_float]] [[dbg_int]]

target triple = "spirv64-unknown-unknown"

define spir_func void @multi_params(i32 %a, float %b, i32 %c) !dbg !10 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-type-function-multi-scalar-params.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{null, !8, !9, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

!10 = distinct !DISubprogram(name: "multi_params", linkageName: "multi_params", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
