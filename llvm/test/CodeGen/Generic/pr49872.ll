; REQUIRES: powerpc-registered-target
; RUN: llc -mtriple powerpc64le-unknown-linux-gnu -filetype=obj < %s | \
; RUN:   llvm-dwarfdump -debug-info - | FileCheck %s

; PR49872
; Check an integer constant for Fortran CHARACTER(1) will not cause a crash in
; DebugHandlerBase::isUnsignedDIType when calling DwarfUnit::addConstantValue

; CHECK:             DW_TAG_formal_parameter
; CHECK:                 DW_AT_const_value     (122)
; CHECK:                 DW_AT_name    ("arg")
; CHECK:                 DW_AT_type    ([[TYPE:[a-z0-9]+]] "char string")
; CHECK: [[TYPE]]:   DW_TAG_string_type
; CHECK:                 DW_AT_name      ("char string")
; CHECK:                 DW_AT_byte_size (0x01)

source_filename = "1.ll"

define dso_local void @s1_after_sroa([1 x i8] %arg) local_unnamed_addr #0 !dbg !5 {
s1_entry:
  call void @llvm.dbg.value(metadata i8 122, metadata !10, metadata !DIExpression()), !dbg !11
  ret void, !dbg !12
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #5

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 3}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, splitDebugInlining: false)
!3 = !DIFile(filename: "1.f", directory: ".")
!4 = !{}
!5 = distinct !DISubprogram(name: "s1", linkageName: "s1", scope: !3, file: !3, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void")
!9 = !DIStringType(name: "char string", size: 8)
!10 = !DILocalVariable(name: "arg", arg: 1, scope: !5, file: !3, type: !9)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocation(line: 4, scope: !5)
!13 = !DILocation(line: 5, scope: !5)
!14 = distinct !DISubprogram(name: "p", linkageName: "main", scope: !3, file: !3, line: 7, type: !15, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{!17}
!17 = !DIBasicType(name: "INTEGER", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 8, scope: !14)
