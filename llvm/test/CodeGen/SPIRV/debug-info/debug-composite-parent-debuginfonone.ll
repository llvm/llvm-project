; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=FWD
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VAL %}
; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=0 -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=0 -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; struct S is declared in f, and f's parameter type is S*. f is scoped in an
; enum, and enums are not lowered yet, so f fails.
;
; The typedef T only makes emission start at f instead of S. T is declared in
; f and is a retained type, so it is emitted first and opens f. f's parameter
; type S* then opens S, and S's Parent is f, which is still open, so S is
; emitted with f's reserved id. f then fails, and that id is defined as
; DebugInfoNone. If S were emitted first instead, f would fail while S is
; still open, and S would not be emitted at all.
;
; -spirv-debug-scope-forward-refs defaults to off, so without that flag the
; back edge is dropped and S is not emitted.
;
; NonSemantic.Shader.DebugInfo says DebugTypeComposite's Parent must be a
; DebugCompilationUnit, DebugFunction, DebugLexicalBlock, or
; DebugTypeComposite. DebugInfoNone is not one of those. spirv-val rejects
; it, and that matches the spec.

; CHECK-SPIRV-NOT: DebugTypeComposite
; CHECK-SPIRV-NOT: OpExtInstWithForwardRefsKHR

; FWD: OpExtension "SPV_KHR_relaxed_extended_instruction"
; FWD: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; FWD-DAG: [[void:%[0-9]+]] = OpTypeVoid
; FWD-DAG: [[name_s:%[0-9]+]] = OpString "S"
; FWD: [[s:%[0-9]+]] = OpExtInstWithForwardRefsKHR [[void]] [[ext]] DebugTypeComposite [[name_s]] {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} [[parent:%[0-9]+]]
; FWD: [[parent]] = OpExtInst [[void]] [[ext]] DebugInfoNone

; VAL: DebugTypeComposite: expected operand Parent must be a result id of a lexical scope

define spir_func void @f(ptr addrspace(4) %p) !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "composite-parent.cpp", directory: "/src")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", scope: !6, file: !3, line: 3, baseType: !12)
!6 = distinct !DISubprogram(name: "f", scope: !7, file: !3, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !3, line: 1, size: 32, elements: !9)
!8 = !DISubroutineType(types: !10)
!9 = !{}
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, dwarfAddressSpace: 4)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !6, file: !3, line: 2, size: 32, elements: !9)
