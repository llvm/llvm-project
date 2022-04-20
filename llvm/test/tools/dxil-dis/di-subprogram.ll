; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-unknown"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.used = !{!5}
!llvm.lines = !{!13, !14, !15, !16}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
; CHECK: !1 = !DIFile(filename: "some-source", directory: "some-path")
!1 = !DIFile(filename: "some-source", directory: "some-path")
!2 = !{}

; CHECK: !3 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 7, !"Dwarf Version", i32 2}
; CHECK: !4 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: !5 = distinct !DISubprogram(name: "fma", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, function: !0, variables: !9)
!5 = distinct !DISubprogram(name: "fma", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)

; CHECK: !6 = !DISubroutineType(types: !7)
!6 = !DISubroutineType(types: !7)

; CHECK: !7 = !{!8, !8, !8, !8}
!7 = !{!8, !8, !8, !8}

; CHECK: !8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

; CHECK: !9 = !{!10, !11, !12}
!9 = !{!10, !11, !12}

; CHECK: !10 = !DILocalVariable(tag: DW_TAG_variable, name: "x", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!10 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 1, type: !8)

; CHECK: !11 = !DILocalVariable(tag: DW_TAG_variable, name: "y", arg: 2, scope: !5, file: !1, line: 1, type: !8)
!11 = !DILocalVariable(name: "y", arg: 2, scope: !5, file: !1, line: 1, type: !8)

; CHECK: !12 = !DILocalVariable(tag: DW_TAG_variable, name: "z", arg: 3, scope: !5, file: !1, line: 1, type: !8)
!12 = !DILocalVariable(name: "z", arg: 3, scope: !5, file: !1, line: 1, type: !8)


; CHECK: !13 = !DILocation(line: 0, scope: !5)
; CHECK: !14 = !DILocation(line: 2, column: 12, scope: !5)
; CHECK: !15 = !DILocation(line: 2, column: 16, scope: !5)
; CHECK: !16 = !DILocation(line: 2, column: 3, scope: !5)

!13 = !DILocation(line: 0, scope: !5)
!14 = !DILocation(line: 2, column: 12, scope: !5)
!15 = !DILocation(line: 2, column: 16, scope: !5)
!16 = !DILocation(line: 2, column: 3, scope: !5)
