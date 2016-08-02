; RUN: %llc_dwarf -O0 < %s | FileCheck %s -check-prefix ARGUMENT
; RUN: %llc_dwarf -O0 < %s | FileCheck %s -check-prefix VARIABLE
; PR 13202

define i32 @main() uwtable !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !18, metadata !DIExpression()), !dbg !21
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !22, metadata !DIExpression()), !dbg !23
  tail call void @smth(i32 0), !dbg !24
  tail call void @smth(i32 0), !dbg !25
  ret i32 0, !dbg !19
}

declare void @smth(i32)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 (trunk 159419)", isOptimized: true, emissionKind: 0, file: !26, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports:  !2)
!1 = !{i32 0}
!2 = !{}
!3 = !{!5, !10}
!5 = distinct !DISubprogram(name: "main", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 10, file: !26, scope: !6, type: !7, variables: !2)
!6 = !DIFile(filename: "inline-bug.cc", directory: "/tmp/dbginfo/pr13202")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "f", linkageName: "_ZL1fi", line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !26, scope: !6, type: !11, variables: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{!9, !9}
!13 = !{!15, !16}
!15 = !DILocalVariable(name: "argument", line: 3, arg: 1, scope: !10, file: !6, type: !9)

; Two DW_TAG_formal_parameter: one abstract and one inlined.
; ARGUMENT: {{.*Abbrev.*DW_TAG_formal_parameter}}
; ARGUMENT: {{.*Abbrev.*DW_TAG_formal_parameter}}
; ARGUMENT-NOT: {{.*Abbrev.*DW_TAG_formal_parameter}}

!16 = !DILocalVariable(name: "local", line: 4, scope: !10, file: !6, type: !9)

; Two DW_TAG_variable: one abstract and one inlined.
; VARIABLE: {{.*Abbrev.*DW_TAG_variable}}
; VARIABLE: {{.*Abbrev.*DW_TAG_variable}}
; VARIABLE-NOT: {{.*Abbrev.*DW_TAG_variable}}

!18 = !DILocalVariable(name: "argument", line: 3, arg: 1, scope: !10, file: !6, type: !9)
!19 = !DILocation(line: 11, column: 10, scope: !5)
!21 = !DILocation(line: 3, column: 25, scope: !10, inlinedAt: !19)
!22 = !DILocalVariable(name: "local", line: 4, scope: !10, file: !6, type: !9)
!23 = !DILocation(line: 4, column: 16, scope: !10, inlinedAt: !19)
!24 = !DILocation(line: 5, column: 3, scope: !10, inlinedAt: !19)
!25 = !DILocation(line: 6, column: 3, scope: !10, inlinedAt: !19)
!26 = !DIFile(filename: "inline-bug.cc", directory: "/tmp/dbginfo/pr13202")
!27 = !{i32 1, !"Debug Info Version", i32 3}
