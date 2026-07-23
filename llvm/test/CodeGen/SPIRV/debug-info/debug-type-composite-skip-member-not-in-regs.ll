; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypeMember
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A member whose type is not in DebugTypeRegs is skipped rather than referenced
; with a dangling id. The one member here has a pointer type with no DWARF
; address space, which emitDebugTypePointer skips for lack of a storage class,
; so no DebugTypeMember is emitted. The composite itself is still emitted, with
; no members. The skip is orthogonal to later type support.

; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugTypeComposite

define spir_func void @test() !dbg !13 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "composite.hlsl", directory: "/src")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "K", file: !3, line: 1, size: 64, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "p", file: !3, line: 2, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 10, type: !14, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DISubroutineType(types: !15)
!15 = !{null}
