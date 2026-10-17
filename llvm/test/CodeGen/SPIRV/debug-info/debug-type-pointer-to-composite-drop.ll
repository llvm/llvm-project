; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypePointer
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A pointer whose pointee is a composite is dropped. Pointers are emitted before
; composites, so the pointee has no id yet when emitDebugTypePointer runs.
; The composite itself is still emitted. Emitting the DebugType*
; nodes in dependency order, tracked in
; https://github.com/llvm/llvm-project/issues/211850, would emit the
; DebugTypePointer and change the expected output of this test

; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: DebugTypeComposite

define spir_func void @test() !dbg !11 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "pointer-to-struct.hlsl", directory: "/src")

; S *p. Only the pointer is retained; S is reached through the pointee.
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, dwarfAddressSpace: 4)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !3, line: 2, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
