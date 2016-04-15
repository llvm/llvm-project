; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -regalloc=basic < %s | FileCheck %s
; Test to check separate label for inlined function argument.

define i32 @foo(i32 %y) nounwind optsize ssp !dbg !1 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !0, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  %0 = tail call i32 (...) @zoo(i32 %y) nounwind, !dbg !9 ; <i32> [#uses=1]
  ret i32 %0, !dbg !9
}

declare i32 @zoo(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar(i32 %x) nounwind optsize ssp !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !8)
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !0, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !1, inlinedAt: !DILocation(scope: !8))
  %0 = tail call i32 (...) @zoo(i32 1) nounwind, !dbg !12 ; <i32> [#uses=1]
  %1 = add nsw i32 %0, %x, !dbg !13               ; <i32> [#uses=1]
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = !DILocalVariable(name: "y", line: 2, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !3, scopeLine: 2, file: !18, scope: !2, type: !4, variables: !15)
!2 = !DIFile(filename: "f.c", directory: "/tmp")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !18, enums: !19, retainedTypes: !19, imports:  null)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DILocalVariable(name: "x", line: 6, arg: 1, scope: !8, file: !2, type: !6)
!8 = distinct !DISubprogram(name: "bar", linkageName: "bar", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !3, scopeLine: 6, file: !18, scope: !2, type: !4, variables: !16)
!9 = !DILocation(line: 3, scope: !10)
!10 = distinct !DILexicalBlock(line: 2, column: 0, file: !18, scope: !1)
!11 = !{i32 1}
!12 = !DILocation(line: 3, scope: !10, inlinedAt: !13)
!13 = !DILocation(line: 7, scope: !14)
!14 = distinct !DILexicalBlock(line: 6, column: 0, file: !18, scope: !8)
!15 = !{!0}
!16 = !{!7}
!18 = !DIFile(filename: "f.c", directory: "/tmp")
!19 = !{}

;CHECK: DEBUG_VALUE: bar:x <- %E
;CHECK: Ltmp
;CHECK:	DEBUG_VALUE: foo:y <- 1{{$}}
!20 = !{i32 1, !"Debug Info Version", i32 3}
