; RUN: llc --filetype=obj %s -o %t.dxbc
; RUN: llvm-objcopy --dump-section=ILDB=%t.bc %t.dxbc
; RUN: dxil-dis %t.bc -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

define void @sr() !dbg !7 {
  unreachable
}

; CHECK-DAG: !llvm.dbg.cu = !{[[CU:![0-9]+]]}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6}

; CHECK-DAG: [[CU]] = distinct !DICompileUnit(language: DW_LANG_Ada95, file: [[F:![0-9]+]], producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: 1, retainedTypes: [[TS:![0-9]+]], subprograms: [[SPS:![0-9]+]])
; CHECK-DAG: [[F]] = !DIFile(filename: "subrange_type.adb", directory: "/dir")
; CHECK-DAG: [[TS]] = !{[[T:![0-9]+]]}
; CHECK-DAG: [[T]] = !DIBasicType(name: "sr__Tint_rangeB", size: 32, encoding: DW_ATE_signed)
; CHECK-DAG: [[SPS]] = !{[[SP:![0-9]+]]}
; CHECK-DAG: [[SP]] = !DISubprogram(name: "sr", scope: !1, file: !1, line: 1, type: [[ST:![0-9]+]], isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, function: void ()* @sr, variables: [[VS:![0-9]+]])
; CHECK-DAG: [[ST]] = !DISubroutineType
; CHECK-DAG: [[VS]] = !{[[V:![0-9]+]]}
; CHECK-DAG: [[V]] = !DILocalVariable(tag: DW_TAG_auto_variable, name: "x", scope: [[SP]], file: [[F]], line: 3, type: [[T]])

!0 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !1, producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2)
!1 = !DIFile(filename: "subrange_type.adb", directory: "/dir")
!2 = !{!3}
!3 = !DISubrangeType(name: "sr__int_range", file: !1, line: 2, size: 32, align: 32, baseType: !4, lowerBound: i64 -7, upperBound: i64 23)
!4 = !DIBasicType(name: "sr__Tint_rangeB", size: 32, encoding: DW_ATE_signed)
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = distinct !DISubprogram(name: "sr", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 3, type: !3, align: 32)
