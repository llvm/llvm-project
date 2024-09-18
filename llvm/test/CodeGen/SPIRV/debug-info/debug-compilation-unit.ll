; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION 
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR-DAG: [[type_void:%[0-9]+\:type]] = OpTypeVoid
; CHECK-MIR-DAG: [[type_i64:%[0-9]+\:type\(s64\)]] = OpTypeInt 32, 0
; CHECK-MIR-DAG: [[dwarf_version:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 5
; CHECK-MIR-DAG: [[source_language:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 3
; CHECK-MIR-DAG: [[debug_info_version:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 21
; CHECK-MIR-DAG: [[filename_str:%[0-9]+\:id\(s32\)]] = OpString 1094795567, 1094795585, 792805697, 1111638594, 1111638594, 1128481583, 1128481603, {{1697596227|1700545347}}, 1886216568, 1663985004, 0
; CHECK-MIR-DAG: [[debug_source:%[0-9]+\:id\(s32\)]] = OpExtInst [[type_void]], 3, 35, [[filename_str]]
; CHECK-MIR-DAG: [[debug_compilation_unit:%[0-9]+\:id\(s32\)]] = OpExtInst [[type_void]], 3, 1, [[source_language]], [[dwarf_version]], [[debug_source]], [[debug_info_version]]

; CHECK-SPIRV: [[ext_inst_non_semantic:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: [[filename_str:%[0-9]+]] = OpString "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC{{[/\\]}}example.c"
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[dwarf_version:%[0-9]+]] = OpConstant [[type_i32]] 5
; CHECK-SPIRV-DAG: [[debug_info_version:%[0-9]+]] = OpConstant [[type_i32]] 21
; CHECK-SPIRV-DAG: [[source_language:%[0-9]+]] = OpConstant [[type_i32]] 3
; CHECK-SPIRV: [[debug_source:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugSource [[filename_str]]
; CHECK-SPIRV: [[debug_compiation_unit:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugCompilationUnit [[source_language]] [[dwarf_version]] [[debug_source]] [[debug_info_version]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-OPTION-NOT: OpString "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC{{[/\\]}}example.c"

define spir_func void @foo() {
entry:
  ret void
}

define spir_func void @bar() {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
