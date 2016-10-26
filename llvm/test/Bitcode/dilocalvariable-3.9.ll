; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: !9 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 1, type: !10)
; CHECK: !10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.1 (http://llvm.org/git/clang.git c3709e72d22432f53f8e2f14354def31a96734fe)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "main.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.1 (http://llvm.org/git/clang.git c3709e72d22432f53f8e2f14354def31a96734fe)"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIExpression()
!12 = !DILocation(line: 1, column: 16, scope: !6)
!13 = !DILocation(line: 1, column: 19, scope: !6)
