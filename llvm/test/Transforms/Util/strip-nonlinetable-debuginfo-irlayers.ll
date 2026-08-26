; RUN: opt -passes=strip-nonlinetable-debuginfo -S %s | FileCheck %s
;;
;; strip-nonlinetable-debuginfo downgrades -g metadata to line-tables-only. An
;; intermediate-IR layer is line-table data -- a line/column in a DIFile -- so it
;; must survive, and DILayerLoc/DILayerLocList must keep their types rather than
;; being rebuilt as generic MDTuples by the pass's fallback remapping.

define void @f(ptr %p) !dbg !5 {
  store ptr null, ptr %p, align 8, !dbg !20
  ret void, !dbg !21
}

; CHECK: store ptr null, ptr %p, align 8, !dbg ![[DBG:[0-9]+]]
; CHECK-DAG: ![[DBG]] = !DILocation(line: 2, column: 5, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK-DAG: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-DAG: ![[LAYER]] = !DILayerLoc(line: 42, column: 5, file: ![[INTF:[0-9]+]], kind: "tile ir")
; CHECK-DAG: ![[INTF]] = !DIFile(filename: "kernel.tileir", directory: ".")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)

!10 = !DIFile(filename: "kernel.tileir", directory: ".")
!11 = !DILayerLoc(line: 42, column: 5, file: !10, kind: "tile ir")
!12 = !DILayerLocList(!11)

!20 = !DILocation(line: 2, column: 5, scope: !5, irlayers: !12)
!21 = !DILocation(line: 3, column: 1, scope: !5)
