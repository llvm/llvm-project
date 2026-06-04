; RUN: llc --filetype=asm %s -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define float @fmaf(float %x, float %y, float %z) !dbg !4 {
  unreachable
}

declare !dbg !14 double @fma(double %x, double %y, double %z)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.used = !{!5}

; CHECK: DXIL: !0: additional data: !16
; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
; CHECK: !1 = !DIFile(filename: "some-source", directory: "some-path")
!1 = !DIFile(filename: "some-source", directory: "some-path")
!2 = !{}

; CHECK: !3 = !{i32 2, !"Dwarf Version", i32 4}
; CHECK: !4 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: DXIL: !8: to be replaced by: !17
; CHECK: !8 = distinct !DISubprogram(name: "fmaf", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!4 = distinct !DISubprogram(name: "fmaf", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)

; CHECK: !9 = !DISubroutineType(types: !10)
!5 = !DISubroutineType(types: !6)

; CHECK: !10 = !{!11, !11, !11, !11}
!6 = !{!7, !7, !7, !7}

; CHECK: !11 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

; CHECK: !12 = !{!13, !14, !15}
!8 = !{!9, !10, !11}

; CHECK: !13 = !DILocalVariable(name: "x", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 1, type: !7)

; CHECK: !14 = !DILocalVariable(name: "y", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!10 = !DILocalVariable(name: "y", arg: 2, scope: !4, file: !1, line: 1, type: !7)

; CHECK: !15 = !DILocalVariable(name: "z", arg: 3, scope: !8, file: !1, line: 1, type: !11)
!11 = !DILocalVariable(name: "z", arg: 3, scope: !4, file: !1, line: 1, type: !7)

!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}

!14 = !DISubprogram(name: "fma", scope: !1, file: !1, line: 1, type: !15, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !17, !17, !17}
!17 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)

; CHECK: !16 = !{!8}

; CHECK: DXIL: !17: additional data: ptr @fmaf
; CHECK: !17 = !DISubprogram(name: "fmaf", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
