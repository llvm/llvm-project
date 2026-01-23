;; This test checks generation of DISubrangeType.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

;; Test whether DISubrangeType is generated.
; CHECK: !DISubrangeType(name: "sr__int_range", file: !{{[0-9]+}}, line: 2, size: 32, align: 32, baseType: !{{[0-9]+}}, lowerBound: i64 -7, upperBound: i64 23)

; ModuleID = 'subrange_type.ll'
source_filename = "/dir/subrange_type.adb"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "subrange_type.adb", directory: "/dir")
!4 = !{}
!5 = !{!11}
!6 = distinct !DISubprogram(name: "sr", scope: !3, file: !3, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10}
!10 = !DILocalVariable(name: "x", scope: !6, file: !3, line: 3, type: !11, align: 32)
!11 = !DISubrangeType(name: "sr__int_range", file: !3, line: 2, size: 32, align: 32, baseType: !12, lowerBound: i64 -7, upperBound: i64 23)
!12 = !DIBasicType(name: "sr__Tint_rangeB", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 3, column: 4, scope: !6)
!14 = !DILocation(line: 6, column: 5, scope: !6)
