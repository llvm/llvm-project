; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not=DebugTypeFunction --check-prefix=CHECK
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Unsupported composite parameter and pointer without dwarfAddressSpace — DISubroutineType not lowered to DebugTypeFunction.

; CHECK: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK: DebugCompilationUnit
; CHECK-NOT: DebugTypePointer
; CHECK-NOT: DebugTypeFunction

target triple = "spirv64-unknown-unknown"

define spir_func void @opaque_struct_param(i32 %x) !dbg !10 {
entry:
  ret void
}

define spir_func void @ptr_no_as(ptr %p) !dbg !11 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-type-function-omit.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{null, !8}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "opaque_sig", file: !1, line: 50, size: 32, elements: !9, identifier: "opaque_sig")
!9 = !{}

!12 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!10 = distinct !DISubprogram(name: "opaque_struct_param", linkageName: "opaque_struct_param", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = distinct !DISubprogram(name: "ptr_no_as", linkageName: "ptr_no_as", scope: !1, file: !1, line: 2, type: !12, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
