; For http://llvm.org/bugs/show_bug.cgi?id=12942
;   There are two CUs coming from /tmp/foo.c in this module. Make sure it doesn't
;   blow llc up and produces something reasonable.
;

; RUN: %llc_dwarf %s -o %t -filetype=obj -O0
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; ModuleID = 'test.bc'

@str = private unnamed_addr constant [4 x i8] c"FOO\00"
@str1 = private unnamed_addr constant [6 x i8] c"Main!\00"

define void @foo() nounwind !dbg !5 {
entry:
  %puts = tail call i32 @puts(ptr @str), !dbg !23
  ret void, !dbg !25
}

declare i32 @puts(ptr nocapture) nounwind

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, metadata !21, metadata !DIExpression()), !dbg !26
  ; Avoid talking about the pointer size in debug info because that's target dependent
  tail call void @llvm.dbg.value(metadata ptr %argv, metadata !22, metadata !DIExpression(DW_OP_deref, DW_OP_deref)), !dbg !27
  %puts = tail call i32 @puts(ptr @str1), !dbg !28
  tail call void @foo() nounwind, !dbg !30
  ret i32 0, !dbg !31
}

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.2 (trunk 156513)", isOptimized: true, emissionKind: FullDebug, file: !32, enums: !1, retainedTypes: !1, globals: !1, imports: !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !32, scope: !6, type: !7, retainedNodes: !1)
!6 = !DIFile(filename: "foo.c", directory: "/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.2 (trunk 156513)", isOptimized: true, emissionKind: FullDebug, file: !32, enums: !1, retainedTypes: !1, globals: !1, imports: !1)
!12 = distinct !DISubprogram(name: "main", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !9, scopeLine: 11, file: !32, scope: !6, type: !13, retainedNodes: !19)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15, !18}
!15 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!18 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!19 = !{!21, !22}
!21 = !DILocalVariable(name: "argc", line: 11, arg: 1, scope: !12, file: !6, type: !15)
!22 = !DILocalVariable(name: "argv", line: 11, arg: 2, scope: !12, file: !6, type: !18)
!23 = !DILocation(line: 6, column: 3, scope: !24)
!24 = distinct !DILexicalBlock(line: 5, column: 16, file: !32, scope: !5)
!25 = !DILocation(line: 7, column: 1, scope: !24)
!26 = !DILocation(line: 11, column: 14, scope: !12)
!27 = !DILocation(line: 11, column: 26, scope: !12)
!28 = !DILocation(line: 12, column: 3, scope: !29)
!29 = distinct !DILexicalBlock(line: 11, column: 34, file: !32, scope: !12)
!30 = !DILocation(line: 13, column: 3, scope: !29)
!31 = !DILocation(line: 14, column: 3, scope: !29)
!32 = !DIFile(filename: "foo.c", directory: "/tmp")

; This test is simple to be cross platform (many targets don't yet have
; sufficiently good DWARF emission and/or dumping)
; CHECK: {{DW_TAG_compile_unit}}
; CHECK: {{foo\.c}}

!33 = !{i32 1, !"Debug Info Version", i32 3}
