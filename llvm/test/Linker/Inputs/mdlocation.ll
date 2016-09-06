define void @foo1() !dbg !0 {
  ret void, !dbg !3
}

!named = !{!1, !2, !3, !4, !5}

!0 = distinct !DISubprogram(file: !7, scope: !7, line: 1, name: "foo", type: !9, unit: !6)

!1 = !DILocation(line: 3, column: 7, scope: !10)
!2 = !DILocation(line: 3, column: 7, scope: !10, inlinedAt: !1)
!3 = !DILocation(line: 3, column: 7, scope: !10, inlinedAt: !2)
; Test distinct nodes.
!4 = distinct !DILocation(line: 3, column: 7, scope: !10)
!5 = distinct !DILocation(line: 3, column: 7, scope: !10, inlinedAt: !4)

!llvm.dbg.cu = !{!6}
!6 = distinct !DICompileUnit(language: DW_LANG_C89, file: !7)
!7 = !DIFile(filename: "source.c", directory: "/dir")

!llvm.module.flags = !{!8}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !DISubroutineType(types: !{})
!10 = distinct !DILexicalBlock(line: 3, column: 3, file: !7, scope: !0)
