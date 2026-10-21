; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A declaration whose return type is a composite. The composite is emitted
; first, so the subroutine type and the declaration are both emitted.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}tmp{{[/\\]}}skip-type-not-in-regs.c"
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "uses_opaque_sig"
; CHECK-DAG: [[SNAME:%[0-9]+]] = OpString "opaque_only_in_sig"
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS]] [[C0]]
; CHECK: [[C1:%[0-9]+]] = OpConstant [[I32]] 1
; CHECK: [[S:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeComposite [[SNAME]] [[C1]]
; CHECK: [[TF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeFunction [[C0]] [[S]]
; CHECK: [[C2:%[0-9]+]] = OpConstant [[I32]] 2
; CHECK: [[C128:%[0-9]+]] = OpConstant [[I32]] 128
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugFunctionDeclaration [[NAME]] [[TF]] [[DS]] [[C2]] [[C0]] [[CU]] [[NAME]] [[C128]]

target triple = "spirv64-unknown-unknown"

define spir_func void @defined() !dbg !9 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, retainedTypes: !10)
!1 = !DIFile(filename: "skip-type-not-in-regs.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "opaque_only_in_sig", file: !1, line: 1, elements: !12)
!12 = !{}
!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !11)
!11 = !{!8}
!10 = !{!13}

!13 = !DISubprogram(name: "uses_opaque_sig", linkageName: "uses_opaque_sig", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped, spFlags: 0)
!7 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !14)
!14 = !{}
!9 = distinct !DISubprogram(name: "defined", linkageName: "defined", scope: !1, file: !1, line: 10, type: !7, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
