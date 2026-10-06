; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=FWD
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; struct S { int x; struct S *next; };
;
; Without SPV_KHR_relaxed_extended_instruction the back edge is in progress
; while S is open, so S lists only member x. Once S and the pointer exist, the
; later walk emits member next. With the extension, the pointer is
; OpExtInstWithForwardRefsKHR naming S, and both members are emitted.

; CHECK-SPIRV: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[str_S:%[0-9]+]] = OpString "S"
; CHECK-SPIRV-DAG: [[str_x:%[0-9]+]] = OpString "x"
; CHECK-SPIRV-DAG: [[str_next:%[0-9]+]] = OpString "next"
; CHECK-SPIRV: [[mem_x:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_x]]
; CHECK-SPIRV-NOT: DebugTypeMember [[str_next]]
; CHECK-SPIRV: [[comp:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeComposite [[str_S]] {{.*}} [[mem_x]]{{$}}
; CHECK-SPIRV: OpExtInst [[void]] [[ext]] DebugTypePointer [[comp]]
; CHECK-SPIRV: OpExtInst [[void]] [[ext]] DebugTypeMember [[str_next]]

; FWD: OpExtension "SPV_KHR_relaxed_extended_instruction"
; FWD: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; FWD-DAG: [[void:%[0-9]+]] = OpTypeVoid
; FWD-DAG: [[str_S:%[0-9]+]] = OpString "S"
; FWD-DAG: [[str_next:%[0-9]+]] = OpString "next"
; FWD: [[ptr:%[0-9]+]] = OpExtInstWithForwardRefsKHR [[void]] [[ext]] DebugTypePointer [[comp:%[0-9]+]]
; FWD: [[mem_next:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugTypeMember [[str_next]] [[ptr]]
; FWD: [[comp]] = OpExtInst [[void]] [[ext]] DebugTypeComposite [[str_S]] {{.*}} [[mem_next]]

define spir_func void @test() !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "self-pointer.c", directory: "/src")

!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 128, elements: !11)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{null}
!11 = !{!12, !13}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !3, line: 2, baseType: !9, size: 32)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "next", file: !3, line: 3, baseType: !14, size: 64, offset: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, dwarfAddressSpace: 4)
