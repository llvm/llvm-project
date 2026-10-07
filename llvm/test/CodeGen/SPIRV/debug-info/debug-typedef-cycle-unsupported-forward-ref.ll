; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=FWD
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Typedef T whose base is a pointer back to T, scoped in an enumeration type.
; The enumeration is unsupported, so T fails after the pointer has already named
; T's forward id. With SPV_KHR_relaxed_extended_instruction and
; -spirv-debug-scope-forward-refs that id is defined
; as DebugInfoNone; without it both edges are dropped.

; CHECK-SPIRV-NOT: DebugTypedef
; CHECK-SPIRV-NOT: DebugTypePointer
; CHECK-SPIRV-NOT: OpExtInstWithForwardRefsKHR

; FWD: OpExtension "SPV_KHR_relaxed_extended_instruction"
; FWD: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; FWD-DAG: [[void:%[0-9]+]] = OpTypeVoid
; FWD: OpExtInstWithForwardRefsKHR [[void]] [[ext]] DebugTypePointer [[none:%[0-9]+]]
; FWD: [[none]] = OpExtInst [[void]] [[ext]] DebugInfoNone
; FWD-NOT: DebugTypedef

define spir_func void @test() !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "typedef-cycle.c", directory: "/src")

!4 = !{!5, !8}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", file: !3, line: 2, baseType: !8, scope: !7)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !3, line: 1, size: 32, elements: !12)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, dwarfAddressSpace: 4)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{}
