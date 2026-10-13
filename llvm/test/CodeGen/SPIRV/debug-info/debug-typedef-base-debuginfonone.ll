; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=FWD
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VAL %}

; typedef U whose base is T, and T whose base is U. T is scoped in an
; unsupported enumeration.
;
; With SPV_KHR_relaxed_extended_instruction and
; -spirv-debug-scope-forward-refs, U is emitted while T is open, so
; U's Base Type is T's reserved id. T then fails, and that id is defined as
; DebugInfoNone. Without the extension the back edge is dropped and neither
; typedef is emitted.
;
; NonSemantic.Shader.DebugInfo says DebugTypedef's Base Type is a debugging
; instruction for the named type, and DebugInfoNone is what other instructions
; refer to when the information is unknown. spirv-val still rejects this: it
; requires DebugTypeBasic. That check does not match the spec.

; CHECK-SPIRV-NOT: DebugTypedef
; CHECK-SPIRV-NOT: OpExtInstWithForwardRefsKHR

; FWD: OpExtension "SPV_KHR_relaxed_extended_instruction"
; FWD: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; FWD-DAG: [[void:%[0-9]+]] = OpTypeVoid
; FWD-DAG: [[name_u:%[0-9]+]] = OpString "U"
; FWD: [[u:%[0-9]+]] = OpExtInstWithForwardRefsKHR [[void]] [[ext]] DebugTypedef [[name_u]] [[base:%[0-9]+]]
; FWD: [[base]] = OpExtInst [[void]] [[ext]] DebugInfoNone
; FWD-NOT: DebugTypedef

; VAL: DebugTypedef: expected operand Base Type must be a result id of DebugTypeBasic

define spir_func void @test() !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "typedef-none.c", directory: "/src")

!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", file: !3, line: 2, baseType: !8, scope: !7)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !3, line: 1, size: 32, elements: !12)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "U", file: !3, line: 3, baseType: !5, scope: !3)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{}
