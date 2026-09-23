; Debug instructions are not code, so they must not count towards the XRay
; instruction threshold: whether a function gets instrumented has to be the
; same whether or not the module is built with debug info.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Two real instructions and three debug records, with a threshold of four: the
; function stays uninstrumented.
define i32 @below_threshold(i32 %a) nounwind uwtable "xray-instruction-threshold"="4" !dbg !4 {
entry:
    #dbg_value(i32 %a, !7, !DIExpression(), !10)
    #dbg_value(i32 %a, !8, !DIExpression(), !10)
    #dbg_value(i32 %a, !9, !DIExpression(), !10)
  ret i32 %a, !dbg !10
}

; CHECK-LABEL: below_threshold:
; CHECK-NOT:   xray_sled

; The same function over the threshold is still instrumented.
define i32 @above_threshold(i32 %a) nounwind uwtable "xray-instruction-threshold"="2" !dbg !11 {
entry:
  ret i32 %a, !dbg !12
}

; CHECK-LABEL: above_threshold:
; CHECK:       xray_sled_0:

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "below_threshold", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!13, !13}
!7 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 1, type: !13)
!8 = !DILocalVariable(name: "y", scope: !4, file: !1, line: 1, type: !13)
!9 = !DILocalVariable(name: "z", scope: !4, file: !1, line: 1, type: !13)
!10 = !DILocation(line: 1, column: 1, scope: !4)
!11 = distinct !DISubprogram(name: "above_threshold", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DILocation(line: 2, column: 1, scope: !11)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
