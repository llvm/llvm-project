; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=FWD
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VAL %}
; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=0 -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=0 -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; struct S { int x; } is declared in namespace ns, and ns is declared in f.
; f is scoped in an unsupported enumeration.
;
; Namespaces are emitted before types, so ns is entered first and enters f
; while ns is still open. f's parameter type is S*, and S's Parent is ns, so
; S is emitted with ns's reserved id. f then fails, ns fails, and that id is
; defined as DebugInfoNone. -spirv-debug-scope-forward-refs defaults to off,
; so without that flag the back edge is dropped and S is not emitted.
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

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "ns-parent.cpp", directory: "/src")

!6 = distinct !DISubprogram(name: "f", scope: !7, file: !3, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !3, line: 1, size: 32, elements: !12)
!8 = !DISubroutineType(types: !11)
!11 = !{null, !14}
!12 = !{}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DINamespace(name: "ns", scope: !6)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !13, file: !3, line: 2, size: 32, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !3, line: 2, baseType: !9, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, dwarfAddressSpace: 4)
