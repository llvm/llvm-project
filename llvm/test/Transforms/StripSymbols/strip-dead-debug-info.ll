; RUN: opt -strip-dead-debug-info -verify %s -S | FileCheck %s

; CHECK: ModuleID = '{{.*}}'
; CHECK-NOT: bar
; CHECK-NOT: abcd

@xyz = global i32 2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

; Function Attrs: nounwind readnone ssp
define i32 @fn() #1 !dbg !6 {
entry:
  ret i32 0, !dbg !18
}

; Function Attrs: nounwind readonly ssp
define i32 @foo(i32 %i) #2 !dbg !10 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !15, metadata !DIExpression()), !dbg !20
  %.0 = load i32, i32* @xyz, align 4
  ret i32 %.0, !dbg !21
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone ssp }
attributes #2 = { nounwind readonly ssp }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !{}, retainedTypes: !{}, subprograms: !23, globals: !24)
!1 = !DIFile(filename: "g.c", directory: "/tmp/")
!2 = !{null}
!3 = distinct !DISubprogram(name: "bar", line: 5, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !1, scope: null, type: !4)
!4 = !DISubroutineType(types: !2)
!5 = !DIFile(filename: "g.c", directory: "/tmp/")
!6 = distinct !DISubprogram(name: "fn", linkageName: "fn", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !1, scope: null, type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !1, scope: null, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{!9, !9}
!13 = !DILocalVariable(name: "bb", line: 5, scope: !14, file: !5, type: !9)
!14 = distinct !DILexicalBlock(line: 5, column: 0, file: !1, scope: !3)
!15 = !DILocalVariable(name: "i", line: 7, arg: 1, scope: !10, file: !5, type: !9)
!16 = !DIGlobalVariable(name: "abcd", line: 2, isLocal: true, isDefinition: true, scope: !5, file: !5, type: !9)
!17 = !DIGlobalVariable(name: "xyz", line: 3, isLocal: false, isDefinition: true, scope: !5, file: !5, type: !9, variable: i32* @xyz)
!18 = !DILocation(line: 6, scope: !19)
!19 = distinct !DILexicalBlock(line: 6, column: 0, file: !1, scope: !6)
!20 = !DILocation(line: 7, scope: !10)
!21 = !DILocation(line: 10, scope: !22)
!22 = distinct !DILexicalBlock(line: 7, column: 0, file: !1, scope: !10)
!23 = !{!3, !6, !10}
!24 = !{!16, !17}
!25 = !{i32 1, !"Debug Info Version", i32 3}
