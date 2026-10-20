; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypedef
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; emitDebugTypedef emits nothing when the base type is not in DebugTypeRegs.
; The base type here is a pointer with no DWARF address space, which
; emitDebugTypePointer skips for lack of a storage class. The base type stays
; out of DebugTypeRegs regardless of later type support, so the typedef is
; skipped.

; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"

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
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "MyPtr", file: !3, line: 2, baseType: !7, scope: !3)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
