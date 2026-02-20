; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

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

define spir_func void @foo() !dbg !8 {
entry:
  ret void
}
; CHECK-SPIRV-NOT: Lfunc_end0:

define spir_func void @bar() !dbg !9 {
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
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !10, unit: !0, spFlags: DISPFlagDefinition)
!9 = distinct !DISubprogram(name: "bar", scope: !7, file: !7, line: 1, type: !10, unit: !6, spFlags: DISPFlagDefinition)
!10 = !DISubroutineType(types: !{})
