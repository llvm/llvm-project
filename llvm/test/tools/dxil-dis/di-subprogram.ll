; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define float @fmaf(float %x, float %y, float %z) !dbg !4 {
  unreachable
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.used = !{!5}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
; CHECK: !1 = !DIFile(filename: "some-source", directory: "some-path")
!1 = !DIFile(filename: "some-source", directory: "some-path")
!2 = !{}

; CHECK: !4 = distinct !DISubprogram(name: "fmaf", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped | 536870912, isOptimized: true, function: float (float, float, float)* @fmaf, variables: !8)
!4 = distinct !DISubprogram(name: "fmaf", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)

; CHECK: !5 = !DISubroutineType(types: !6)
!5 = !DISubroutineType(types: !6)

; CHECK: !6 = !{!7, !7, !7, !7}
!6 = !{!7, !7, !7, !7}

; CHECK: !7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

; CHECK: !8 = !{!9, !10, !11}
!8 = !{!9, !10, !11}

; CHECK: !9 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "x", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 1, type: !7)

; CHECK: !10 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "y", arg: 2, scope: !4, file: !1, line: 1, type: !7)
!10 = !DILocalVariable(name: "y", arg: 2, scope: !4, file: !1, line: 1, type: !7)

; CHECK: !11 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "z", arg: 3, scope: !4, file: !1, line: 1, type: !7)
!11 = !DILocalVariable(name: "z", arg: 3, scope: !4, file: !1, line: 1, type: !7)

; CHECK: !12 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Dwarf Version", i32 4}
; CHECK: !13 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 2, !"Debug Info Version", i32 3}
