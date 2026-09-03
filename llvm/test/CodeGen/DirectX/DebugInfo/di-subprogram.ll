; RUN: opt -S -passes=dxil-debug-info %s -o - | FileCheck %s
; RUN: llc %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-COMMENT
target triple = "dxil-unknown-shadermodel6.7-library"

define float @fmaf(float %x, float %y, float %z) !dbg !4 {
  unreachable
}

declare !dbg !14 double @fma(double %x, double %y, double %z)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.used = !{!5}

; CHECK-DAG: [[CU:![0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C99, file: [[FILE:![0-9]+]], producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !{{[0-9]+}}, splitDebugInlining: false, nameTableKind: None)
; CHECK-COMMENT-DAG: DXIL: [[CU]]: additional data: [[SUBPROGRAMS:![0-9]+]]
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
; CHECK-DAG: [[FILE]] = !DIFile(filename: "some-source", directory: "some-path")
!1 = !DIFile(filename: "some-source", directory: "some-path")
!2 = !{}

; CHECK-DAG: !{i32 2, !"Dwarf Version", i32 4}
; CHECK-DAG: !{i32 2, !"Debug Info Version", i32 3}

; CHECK-DAG: [[SP:![0-9]+]] = distinct !DISubprogram(name: "fmaf", scope: [[FILE]], file: [[FILE]], line: 1, type: [[SPTY:![0-9]+]], scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: [[CU]], retainedNodes: [[VARS:![0-9]+]])
; CHECK-COMMENT-DAG: DXIL: [[SP]]: to be replaced by: [[NEWSP:![0-9]+]]
!4 = distinct !DISubprogram(name: "fmaf", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)

; CHECK-DAG: [[SPTY]] = !DISubroutineType(types: [[SPTYPES:![0-9]+]])
!5 = !DISubroutineType(types: !6)

; CHECK-DAG: [[SPTYPES]] = !{[[FLOAT:![0-9]+]], [[FLOAT]], [[FLOAT]], [[FLOAT]]}
!6 = !{!7, !7, !7, !7}

; CHECK-DAG: [[FLOAT]] = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

; CHECK-DAG: [[VARS]] = !{[[X:![0-9]+]], [[Y:![0-9]+]], [[Z:![0-9]+]]}
!8 = !{!9, !10, !11}

; CHECK-DAG: [[X]] = !DILocalVariable(name: "x", arg: 1, scope: [[SP]], file: [[FILE]], line: 1, type: [[FLOAT]])
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 1, type: !7)

; CHECK-DAG: [[Y]] = !DILocalVariable(name: "y", arg: 2, scope: [[SP]], file: [[FILE]], line: 1, type: [[FLOAT]])
!10 = !DILocalVariable(name: "y", arg: 2, scope: !4, file: !1, line: 1, type: !7)

; CHECK-DAG: [[Z]] = !DILocalVariable(name: "z", arg: 3, scope: [[SP]], file: [[FILE]], line: 1, type: [[FLOAT]])
!11 = !DILocalVariable(name: "z", arg: 3, scope: !4, file: !1, line: 1, type: !7)

!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}

!14 = !DISubprogram(name: "fma", scope: !1, file: !1, line: 1, type: !15, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !17, !17, !17}
!17 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)

; CHECK-COMMENT-DAG: [[SUBPROGRAMS]] = !{[[SP]]}
; CHECK-COMMENT-DAG: DXIL: [[NEWSP]]: additional data: ptr @fmaf
; CHECK-COMMENT-DAG: [[NEWSP]] = !DISubprogram(name: "fmaf", scope: [[FILE]], file: [[FILE]], line: 1, type: [[SPTY]], scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: [[CU]], retainedNodes: [[VARS]])
