; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION 
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR-DAG: [[type_void:%[0-9]+\:type]] = OpTypeVoid
; CHECK-MIR-DAG: [[type_i64:%[0-9]+\:type\(s64\)]] = OpTypeInt 32, 0
; CHECK-MIR-DAG: [[dwarf_version:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 5
; CHECK-MIR-DAG: [[debug_info_version:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 3
; CHECK-MIR-DAG: [[source_language_sycl:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 7
; CHECK-MIR-DAG: [[source_language_cpp:%[0-9]+\:iid\(s32\)]] = OpConstantI [[type_i64]], 4
; CHECK-MIR-DAG: [[filename_str_sycl:%[0-9]+\:id\(s32\)]] = OpString 1094795567, 1094795585, 792805697, 1111638594, 1111638594, 1128481583, 1128481603, {{1697596227|1700545347}}, 1886216568, 1663985004, 0
; CHECK-MIR-DAG: [[filename_str_cpp:%[0-9]+\:id\(s32\)]] = OpString 1145324591, 1145324612, 793003076, 1162167621, 1162167621, 1179010607, 1179010630, 1697596998, 1886216568, 774989164, 7368803
; CHECK-MIR-DAG: [[debug_source_sycl:%[0-9]+\:id\(s32\)]] = OpExtInst [[type_void]], 3, 35, [[filename_str_sycl]]
; CHECK-MIR-DAG: OpExtInst [[type_void]], 3, 1, [[debug_info_version]], [[dwarf_version]], [[debug_source_sycl]], [[source_language_sycl]]
; CHECK-MIR-DAG: [[debug_source_cpp:%[0-9]+\:id\(s32\)]] = OpExtInst [[type_void]], 3, 35, [[filename_str_cpp]]
; CHECK-MIR-DAG: OpExtInst [[type_void]], 3, 1, [[debug_info_version]], [[dwarf_version]], [[debug_source_cpp]], [[source_language_cpp]]

; CHECK-SPIRV: [[ext_inst_non_semantic:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: [[filename_str_sycl:%[0-9]+]] = OpString "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC{{[/\\]}}example.c"
; CHECK-SPIRV: [[filename_str_cpp:%[0-9]+]] = OpString "/DDDDDDDDDD/EEEEEEEE/FFFFFFFFF{{[/\\]}}example1.cpp"
; CHECK-SPIRV-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[type_i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[dwarf_version:%[0-9]+]] = OpConstant [[type_i32]] 5
; CHECK-SPIRV-DAG: [[source_language_sycl:%[0-9]+]] = OpConstant [[type_i32]] 7
; CHECK-SPIRV-DAG: [[source_language_cpp:%[0-9]+]] = OpConstant [[type_i32]] 4
; CHECK-SPIRV-DAG: [[debug_info_version:%[0-9]+]] = OpConstant [[type_i32]] 3
; CHECK-SPIRV: [[debug_source_sycl:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugSource [[filename_str_sycl]]
; CHECK-SPIRV: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugCompilationUnit [[debug_info_version]] [[dwarf_version]] [[debug_source_sycl]] [[source_language_sycl]]
; CHECK-SPIRV: [[debug_source_cpp:%[0-9]+]] = OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugSource [[filename_str_cpp]]
; CHECK-SPIRV: OpExtInst [[type_void]] [[ext_inst_non_semantic]] DebugCompilationUnit [[debug_info_version]] [[dwarf_version]] [[debug_source_cpp]] [[source_language_cpp]]

; CHECK-OPTION-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-OPTION-NOT: OpString "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC{{[/\\]}}example.c"

define spir_func void @foo() {
entry:
  ret void
}
; CHECK-SPIRV-NOT: Lfunc_end0:

define spir_func void @bar() {
entry:
  ret void
}
; CHECK-SPIRV-NOT: Lfunc_end1:

!llvm.dbg.cu = !{!0, !6}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_SYCL, file: !1, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = distinct !DICompileUnit(language: DW_LANG_OpenCL_CPP, file: !7, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!7 = !DIFile(filename: "example1.cpp", directory: "/DDDDDDDDDD/EEEEEEEE/FFFFFFFFF", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
