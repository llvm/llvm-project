; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: [[ext_inst_non_semantic:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[flag_zero:%[0-9]+]] = OpConstant [[type_int32]] 0{{$}}
; CHECK-SPIRV-DAG: OpString "float"
; CHECK-SPIRV-DAG: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugTypeFunction [[flag_zero]]

target triple = "spirv64-unknown-unknown"

define spir_func float @float_sig() !dbg !9 {
entry:
  ret float 0.000000e+00
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-type-function-scalar-returns.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

!9 = distinct !DISubprogram(name: "float_sig", linkageName: "float_sig", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
